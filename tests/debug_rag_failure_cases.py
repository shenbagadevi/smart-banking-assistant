import logging
from src.api.v1.agents.nodes.router_node import router_node
from src.api.v1.agents.nodes.vector_node import vector_search_node
from src.api.v1.agents.nodes import generate_answer_node as gen_mod
from src.api.v1.agents.nodes import evaluate_answer_node as eval_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_rag_failure_cases")


# fake docs
class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


fake_docs = [FakeDoc("Rate 7%", {"document_name": "HomeLoanPolicy", "chunk_id": 1})]

# patch hybrid_search used by vector_node
import src.api.v1.agents.nodes.vector_node as vector_node_mod

vector_node_mod.hybrid_search = lambda q, vector_k=20, fts_k=20, final_k=5: fake_docs

state = {"query": "What is the interest rate for a home loan?"}
state = router_node(state)
state = vector_search_node(state)

print("\n--- Scenario A: generation fails, evaluator succeeds ---")


# patch generate's local _get_llm to raise on invoke
class GenFailLLM:
    def invoke(self, messages):
        raise Exception("simulated generation failure")


gen_mod._get_llm = lambda: GenFailLLM()


# patch evaluate's local _get_llm to return NO (verdict)
class EvalLLM:
    def invoke(self, messages):
        class Resp:
            def __init__(self, content):
                self.content = content

        return Resp("NO")


eval_mod._get_llm = lambda: EvalLLM()

s_after_gen = gen_mod.generate_answer_node(state)
print("after generate keys:", list(s_after_gen.keys()))
print("response:", s_after_gen.get("response"))
print("confidence_score:", s_after_gen.get("confidence_score"))

s_after_eval = eval_mod.evaluate_answer_node(s_after_gen)
print("after evaluate is_valid:", s_after_eval.get("is_valid"))
print("after evaluate retry_count:", s_after_eval.get("retry_count"))
print("after evaluate confidence_score:", s_after_eval.get("confidence_score"))

print("\n--- Scenario B: generation fails, evaluator also fails ---")


# patch evaluate's _get_llm to raise
class EvalFailLLM:
    def invoke(self, messages):
        raise Exception("simulated eval failure")


eval_mod._get_llm = lambda: EvalFailLLM()

s_after_gen = gen_mod.generate_answer_node(state)
print("after generate response:", s_after_gen.get("response"))

s_after_eval = eval_mod.evaluate_answer_node(s_after_gen)
print("after evaluate retry_count (both failed):", s_after_eval.get("retry_count"))
print(
    "after evaluate confidence_score (both failed):",
    s_after_eval.get("confidence_score"),
)
