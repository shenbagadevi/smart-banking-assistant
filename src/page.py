import streamlit as st
import requests

st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Smart Banking Assistant")

st.write(
    "Ask questions about banking products, loans, accounts, transactions, and policies."
)

query = st.text_area(
    "Enter your question",
    height=100
)

if st.button("Submit"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:

        try:

            response = requests.post(
                "http://localhost:8000/api/v1/query",
                json={"query": query}
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Response Generated")

                st.subheader("Answer")
                st.write(result.get("answer", ""))

                st.subheader("Query Path")
                st.write(result.get("query_path", ""))

                st.subheader("Retry Count")
                st.write(result.get("retry_count", 0))

                st.subheader("Confidence Score")
                st.write(result.get("confidence_score", ""))

                if result.get("sql_query"):
                    st.subheader("Generated SQL")
                    st.code(result["sql_query"], language="sql")

                if result.get("sql_result"):
                    st.subheader("SQL Result")
                    st.json(result["sql_result"])

                if result.get("citations"):
                    st.subheader("Citations")

                    for citation in result["citations"]:
                        st.write(citation)

            else:
                st.error(f"API Error : {response.status_code}")

        except Exception as e:
            st.error(str(e))

            
