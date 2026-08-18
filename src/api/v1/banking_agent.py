import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal

from sentence_transformers import CrossEncoder

from src.api.v1.states.banking_state import BankingState
from src.api.v1.tools.query_rewrite import query_rewriter_node
from src.api.v1.tools.hybrid_search_tool import _search_hybrid
from src.api.v1.schemas.query_schema import AIResponse
from src.core.db import get_sql_database


load_dotenv()


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )


class RouteDecision(BaseModel):
    route: Literal["VECTOR_DB", "RDBMS"]
    reason: str


def router_node(state: BankingState) -> BankingState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a query router for a Smart Banking Assistant.

                Classify the user's query into EXACTLY one of the following routes:

                'VECTOR_DB' - Use this when the query asks about banking policies,
                product brochures, terms and conditions, interest rate tables,
                foreclosure charges, fees, policy rules, FAQs, guidelines, or
                anything that requires reading banking knowledge base documents.

                'RDBMS' - Use this when the query asks about customer accounts,
                transactions, purchase history, loan outstanding amount, next EMI date,
                fixed deposits, credit cards, card transactions, balances, or anything
                answerable from structured banking database tables:
                accounts, transactions, loan_accounts, fixed_deposits,
                credit_cards, card_transactions.

                Reply with the route and one sentence reason.
                """,
            ),
            (
                "human",
                """
                Question:
                {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})

    print(f"[router_node decision]: {decision.route} | reason: {decision.reason}")

    return {
        **state,
        "route": decision.route,
    }


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def nl2sql_node(state: BankingState) -> BankingState:
    print("====== INSIDE nl2sql_node ======")

    llm = _get_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a PostgreSQL expert for a Smart Banking Assistant.

                Given the database schema below, write a single valid SELECT query
                that answers the user's question.

                Rules:
                - Return ONLY the raw SQL.
                - Do not include explanation, markdown, code fences, or backticks.
                - Use only the tables and columns present in the schema.
                - Only SELECT queries are allowed.
                - Do NOT generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE,
                  CREATE, or any DML/DDL statements.
                - Always add LIMIT 100 unless the question asks for aggregate results.
                - For account_id, loan_id, card_id, and fd_id, use exact matching.
                - Do not expose sensitive fields unnecessarily.
                - If mobile is selected, it is already masked in the database.

                Database schema:
                {schema}
                """,
            ),
            (
                "human",
                """
                Question:
                {question}
                """,
            ),
        ]
    )

    sql_chain = sql_prompt | llm

    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "question": state["query"],
        }
    )

    generated_sql = raw_sql.content.strip()

    print("====== GENERATED SQL QUERY ======")
    print(generated_sql)

    if not generated_sql.lower().startswith("select"):
        sql_result = "Unsafe SQL blocked. Only SELECT queries are allowed."
    else:
        try:
            sql_result = db.run(generated_sql)
        except Exception as err:
            sql_result = f"Generated SQL execution error: {err}"

    structured_llm = llm.with_structured_output(AIResponse)

    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful Smart Banking Assistant.

                Answer the user's question using only the SQL query result provided.

                Rules:
                - Be concise and clear.
                - Format amounts, dates, and lists clearly.
                - Do not invent any numbers.
                - Do not use technical jargon.
                - If the SQL result is empty, say no matching records were found.
                - Set policy_citations to "N/A".
                - Set page_no to "N/A".
                - Set document_name to "banking_rdbms".
                - Set query_path to "RDBMS".
                - Set sql_query_executed to the SQL query used.
                - Set retry_count to 0.
                """,
            ),
            (
                "human",
                """
                Question:
                {query}

                SQL Used:
                {sql}

                Query Result:
                {result}
                """,
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm

    answer = nl_chain.invoke(
        {
            "query": state["query"],
            "sql": generated_sql,
            "result": sql_result,
        }
    )

    print("[nl2sql_node] Answer generated.")

    response = answer.model_dump()
    response["query_path"] = "RDBMS"
    response["policy_citations"] = "N/A"
    response["page_no"] = "N/A"
    response["document_name"] = "banking_rdbms"
    response["sql_query_executed"] = generated_sql
    response["retry_count"] = 0

    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response,
    }


def vector_search_node(state: BankingState) -> BankingState:
    print("====== INSIDE vector_search_node: hybrid search ======")

    query = state.get("rewritten_query") or state.get("query")

    results = _search_hybrid(
        query=query,
        k=10,
    )

    print(f"[vector_search_node] Retrieved {len(results)} docs from hybrid search")

    return {
        **state,
        "retrieved_docs": results,
    }


def rerank_node(state: BankingState) -> BankingState:
    print("====== INSIDE rerank_node ======")

    docs = state.get("retrieved_docs", [])

    if not docs:
        print("No retrieved docs found for reranking.")
        return {
            **state,
            "reranked_docs": [],
        }

    query = state.get("rewritten_query") or state.get("query")

    pairs = []

    for doc in docs:
        pairs.append(
            (
                query,
                doc["content"],
            )
        )

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    reranked_docs = []

    for doc, score in ranked_docs[:5]:
        doc["rerank_score"] = float(score)
        reranked_docs.append(doc)

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")

    for i, doc in enumerate(reranked_docs):
        print(
            f"Rank {i + 1} | Score: {doc.get('rerank_score')} | "
            f"Type: {doc.get('metadata', {}).get('content_type', 'unknown')}"
        )

    return {
        **state,
        "reranked_docs": reranked_docs,
    }


def generate_answer_node(state: BankingState) -> BankingState:
    print("====== INSIDE generate_answer_node ======")

    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    reranked_docs = state.get("reranked_docs", [])

    for doc in reranked_docs:
        print("Metadata:", doc.get("metadata", {}))

    context_parts = []

    for doc in reranked_docs:
        metadata = doc.get("metadata", {})

        document_name = metadata.get("document_name", "KB_Smart_Banking.pdf")
        page_number = metadata.get("page_number", metadata.get("page", "unknown"))
        content_type = metadata.get("content_type", metadata.get("chunk_type", "text"))

        context_parts.append(
            f"[Document: {document_name} | "
            f"Page: {page_number} | "
            f"Content Type: {content_type}]\n"
            f"{doc['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a Smart Banking Assistant.

                Answer the user's question using only the provided banking knowledge
                base context.

                Rules:
                - Do not hallucinate policy values, charges, rates, dates, or fees.
                - If the answer is not available in the context, say that the
                  information is not available in the provided knowledge base.
                - Keep the answer clear and business-friendly.
                - Use text, table, and image caption chunks if they are relevant.
                - Cite the source document and page number.
                - Set query_path to "VECTOR_DB".
                - Set sql_query_executed to "N/A".

                Citation rules:
                - document_name: comma-separated list of source documents used.
                - page_no: comma-separated page numbers used.
                - policy_citations: readable citation format like:
                  "KB_Smart_Banking.pdf, Page 3".
                """,
            ),
            (
                "human",
                """
                Context:
                {context}

                Question:
                {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "context": context,
            "query": state["query"],
        }
    )

    print("[generate_answer_node] Answer generated.")

    response = result.model_dump()
    response["query_path"] = "VECTOR_DB"
    response["sql_query_executed"] = "N/A"
    response["retry_count"] = state.get("retry_count", 0)

    return {
        **state,
        "response": response,
    }


def decide_after_vector_search(state: BankingState):
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    print("====== INSIDE decide_after_vector_search ======")
    print("Retrieved docs count:", len(docs))
    print("Retry count:", retry_count)

    if len(docs) > 0:
        return "rerank"

    if retry_count < 2:
        return "query_rewriter"

    return "no_docs_answer"


def no_docs_answer_node(state: BankingState) -> BankingState:
    print("====== INSIDE no_docs_answer_node ======")

    response = {
        "answer": "No relevant documents were found in the banking knowledge base for this query.",
        "policy_citations": "N/A",
        "page_no": "N/A",
        "document_name": "N/A",
        "query_path": "VECTOR_DB",
        "sql_query_executed": "N/A",
        "retry_count": state.get("retry_count", 0),
    }

    return {
        **state,
        "response": response,
    }


def build_banking_graph():
    workflow = StateGraph(BankingState)

    workflow.add_node("router", router_node)
    #check later 
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("no_docs_answer", no_docs_answer_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "VECTOR_DB": "vector_search",
            "RDBMS": "nl2sql",
        },
    )

    workflow.add_conditional_edges(
        "vector_search",
        decide_after_vector_search,
        {
            "rerank": "rerank",
            "query_rewriter": "query_rewriter",
            "no_docs_answer": "no_docs_answer",
        },
    )

    workflow.add_edge("query_rewriter", "vector_search")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", END)

    workflow.add_edge("no_docs_answer", END)
    workflow.add_edge("nl2sql", END)

    banking_agent = workflow.compile()

    graph_image = banking_agent.get_graph().draw_mermaid_png()
    with open("banking_agent.png", "wb") as f:
        f.write(graph_image)

    return banking_agent


banking_graph = build_banking_graph()


def run_banking_agent(query: str):
    print("====== INSIDE run_banking_agent ======")

    initial_state = {
        "query": query,
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "generated_sql": "",
        "sql_result": "",
        "response": {},
        "retry_count": 0,
        "rewritten_query": "",
        "rewritten_queries": [],
    }

    final_state=banking_graph.invoke(initial_state)
    print ("======FINAL STATE========")
    print (final_state)

    response =final_state.get("response")

    if response is None or response =={}:
        return{
            "answer": "No response generated by banking agent",
            "policy_citations": "N/A",
            "page_no":"N/A",
            "document_name":"N/A",
            "query_path":final_state.get("route","UNKNOWN"),
            "sql_query_executed":final_state.get("generated_sql","N/A"),
            "retry_count":final_state.get("retry_count",0)
        }
    return response 

   
