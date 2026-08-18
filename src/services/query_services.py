from src.api.v1.agents.banking_agent import run_banking_agent


def process_query(query: str):
    print("====== INSIDE process_query service ======")

    response = run_banking_agent(query)

    return response
