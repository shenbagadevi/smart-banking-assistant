import logging
import os

logger = logging.getLogger(__name__)
client = None


def rerank_documents(query, documents):
    """
    Reorders retrieved chunks using Cohere.
    """
    try:
        if not documents:
            return []

        # Lazy import to avoid import-time failure when cohere isn't installed
        global client
        if client is None:
            try:
                import cohere

                client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            except Exception:
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
        # Annotate returned documents with a derived rerank score for debugging
        results = []
        try:
            total = len(response.results)
            for rank, item in enumerate(response.results, start=1):
                doc = documents[item.index]
                # derive a normalized score (1.0 highest -> down)
                score = float(total - (rank - 1)) / float(total)
                try:
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["rerank_score"] = score
                except Exception:
                    pass
                results.append(doc)
        except Exception:
            # Fallback: return documents in original order
            results = documents

        return results
    except Exception:
        logger.exception("Reranking failed")
        return documents
