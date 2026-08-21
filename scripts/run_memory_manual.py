from src.api.v1.services.query_service import process_query

user = "test_user_rose"
print("First: tell name")
r1 = process_query("Hi, I am Rose", user, "corr-rose")
print(r1)
print("Second: who am i")
r2 = process_query("Who am I?", user, "corr-rose")
print(r2)
