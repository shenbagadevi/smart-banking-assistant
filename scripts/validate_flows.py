from pprint import pprint
from src.api.v1.services.query_service import process_query


def run_case(query, label):
    print(f"\n--- {label} ---")
    resp = process_query(query, user_id="test_user", correlation_id=label)
    pprint(resp)
    return resp


def main():
    cases = [
        ("Hi", "CHAT_HI"),
        ("Who are you?", "CHAT_WHO"),
        ("Thanks", "CHAT_THANKS"),
        (
            "What is the home loan interest rate for women borrowers taking a loan above Rs.1.5 Crore?",
            "RAG_RATE_WOMEN",
        ),
        (
            "Give me last 3 months purchase history of account 1345367",
            "SQL_PURCHASE_HISTORY",
        ),
        (
            "What is my account balance and what home loan interest rate applies for women borrowers?",
            "HYBRID_BALANCE_AND_RATE",
        ),
    ]

    results = {}
    for q, label in cases:
        results[label] = run_case(q, label)

    print("\nSummary: \n")
    for k, v in results.items():
        print(k, "->", v.get("query_path"), "retry_count=", v.get("retry_count"))


if __name__ == "__main__":
    main()
