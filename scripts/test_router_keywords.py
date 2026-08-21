q = "Give me last 3 months purchase history of account 1345367".lower()
chat_triggers = (
    "hi",
    "hello",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "who are you",
    "what can you do",
)
sql_triggers = (
    "account",
    "balance",
    "transaction",
    "transactions",
    "purchase history",
    "statement",
    "last month",
    "last",
    "months",
    "debit",
    "credit",
    "transfer",
)
print("query:", q)
print("chat any:", any(t in q for t in chat_triggers))
print("sql any:", any(kw in q for kw in sql_triggers))
print("purchase history in q?", "purchase history" in q)
print("account in q?", "account" in q)
print("last in q?", "last" in q, "month" in q)
