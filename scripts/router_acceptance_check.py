from src.api.v1.agents.nodes.router_node import router_node

queries = [
    "I am Devi",
    "My name is Devi",
    "Hi, I am John",
    "Who are you?",
    "What is my name?",
    "What did I tell you earlier?",
    "What is the current balance of account ID 1345367?",
    "Show the last 5 transactions for account ID 1345367.",
    "What is the total amount credited to account ID 1345367?",
    "What are the foreclosure charges for floating-rate home loans?",
    "What are the eligibility criteria for a home loan?",
]

for q in queries:
    state = {"query": q}
    out = router_node(state)
    print(q, "->", out.get("route"))
