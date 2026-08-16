import logging
import os
import cohere

logger = logging.getLogger(__name__)

try:
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
except Exception:
    client = None


def rerank_documents(query, documents):
    """
    Reorders retrieved chunks using Cohere.
    """
    try:
        if not documents:
            return []
        if client is None:
            logger.warning("Cohere client not configured; returning original docs")
            return documents

        response = client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[f"""
            Product Category:
            {d.metadata.get('product_category')}

            Loan Type:
            {d.metadata.get('loan_type')}

            Section:
            {d.metadata.get('section')}

            Content:
            {d.page_content}
            """ for d in documents],
            top_n=5,
        )

        return [documents[item.index] for item in response.results]
    except Exception:
        logger.exception("Reranking failed")
        return documents
