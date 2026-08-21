from src.api.v1.tools.rag_tool import get_vector_store

store = get_vector_store()

docs = store.similarity_search("home loan foreclosure charges", k=5)

print("RESULT COUNT:", len(docs))

for d in docs:
    print(d.metadata, d.page_content[:200])
