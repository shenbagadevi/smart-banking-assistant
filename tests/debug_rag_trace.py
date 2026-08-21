import logging
from src.api.v1.agents.nodes.router_node import router_node
from src.api.v1.agents.nodes.vector_node import vector_search_node
from src.api.v1.agents.nodes.generate_answer_node import generate_answer_node
from src.api.v1.agents.nodes.evaluate_answer_node import evaluate_answer_node
from src.api.v1.agents import RAGState

# Monkeypatch by assignment
import src.api.v1.tools.hybrid_search_tool as hybrid_tool
import src.api.v1.agents.nodes.node_utils as node_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_rag_trace")


# Fake documents
class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


fake_docs = [
    FakeDoc(
        "Home loan interest rate is 7% per annum as per policy.",
        {"document_name": "HomeLoanPolicy", "chunk_id": 1, "section": "Interest Rates"},
    ),
    FakeDoc(
        "Processing fee of 0.5% applies.",
        {"document_name": "HomeLoanPolicy", "chunk_id": 2, "section": "Fees"},
    ),
]

# Patch hybrid_search used by vector_node module
import src.api.v1.agents.nodes.vector_node as vector_node_mod

vector_node_mod.hybrid_search = lambda q, vector_k=20, fts_k=20, final_k=5: fake_docs

# Prepare a state
state = {
    "query": "What is the interest rate for a home loan?",
}

logger.info("1. Router node")
state = router_node(state)
logger.info("After router: %s", state)

logger.info("2. Vector node")
state = vector_search_node(state)
logger.info("After vector: keys=%s", list(state.keys()))
logger.info("retrieved_docs count=%d", len(state.get("retrieved_docs") or []))


# Patch _get_llm to simulate OpenAI generation failure on first invoke, then a safe evaluator
class FakeLLM:
    def __init__(self):
        self.called = 0

    def invoke(self, messages):
        self.called += 1
        if self.called == 1:
            # Simulate generation failure
            raise Exception("Simulated OpenAI generation failure")

        # For evaluator, return an object with content 'NO' to indicate invalid answer
        class Resp:
            def __init__(self, content):
                self.content = content

        return Resp("NO")


node_utils._get_llm = lambda: FakeLLM()

logger.info("3. Generate answer node (will simulate LLM failure)")
state_after_gen = generate_answer_node(state)
logger.info("After generate: keys=%s", list(state_after_gen.keys()))
logger.info("response=%s", state_after_gen.get("response"))
logger.info("confidence_score=%s", state_after_gen.get("confidence_score"))

logger.info("4. Evaluate answer node")
state_after_eval = evaluate_answer_node(state_after_gen)
logger.info(
    "After evaluate: is_valid=%s should_retry=%s retry_count=%s confidence_score=%s",
    state_after_eval.get("is_valid"),
    state_after_eval.get("should_retry"),
    state_after_eval.get("retry_count"),
    state_after_eval.get("confidence_score"),
)

print("\n=== TRACE OUTPUT ===")
print("router route:", state.get("route"))
print("retrieved_docs count:", len(state.get("retrieved_docs") or []))
print(
    "retrieved_docs[0] metadata:",
    (
        (state.get("retrieved_docs") or [])[0].metadata
        if state.get("retrieved_docs")
        else None
    ),
)
print("generate response:", state_after_gen.get("response"))
print("generate confidence_score:", state_after_gen.get("confidence_score"))
print("evaluate is_valid:", state_after_eval.get("is_valid"))
print("evaluate should_retry:", state_after_eval.get("should_retry"))
print("evaluate retry_count:", state_after_eval.get("retry_count"))
print("evaluate confidence_score:", state_after_eval.get("confidence_score"))
