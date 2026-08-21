from src.api.v1.agents.nodes.router_node import router_node

state = {"query": "Give me last 3 months purchase history of account 1345367"}
new = router_node(state)
print("router output:", new)
