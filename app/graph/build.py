from functools import lru_cache

from anthropic import Anthropic
from langfuse import Langfuse
from langgraph.graph import END, START, StateGraph

from app.config import (
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    FEATURE_SPEC_PATH,
    MODEL_PATH,
    PERSONAS,
    VOYAGE_MODEL,
    VOYAGE_OUTPUT_DIM,
)
from app.graph.nodes.assemble import AssembleNode
from app.graph.nodes.attribute import AttributeNode
from app.graph.nodes.explain import ExplainNode
from app.graph.nodes.predict import PredictNode
from app.graph.state import ClaimState
from app.model.embedder import VoyageEmbedder
from app.model.predictor import ClaimPredictor
from app.observability import get_langfuse
from app.secrets_utils import get_secrets


def build_graph(predictor: ClaimPredictor, anthropic_client: Anthropic, langfuse: Langfuse | None):
    graph = StateGraph(ClaimState)

    graph.add_node("predict", PredictNode(predictor))
    graph.add_node("attribute", AttributeNode(predictor))
    graph.add_node("assemble", AssembleNode())

    graph.add_edge(START, "predict")
    graph.add_edge("predict", "attribute")

    for persona, prompt_path in PERSONAS.items():
        node_name = f"explain_{persona}"
        node = ExplainNode(
            persona=persona,
            prompt_template=prompt_path.read_text(),
            client=anthropic_client,
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            langfuse=langfuse,
        )
        graph.add_node(node_name, node)
        graph.add_edge("attribute", node_name)
        graph.add_edge(node_name, "assemble")

    graph.add_edge("assemble", END)
    return graph.compile()


@lru_cache
def get_predictor() -> ClaimPredictor:
    embedder = VoyageEmbedder(
        api_key=get_secrets()["VOYAGE_API_KEY"],
        model=VOYAGE_MODEL,
        dim=VOYAGE_OUTPUT_DIM,
        langfuse=get_langfuse(),
    )
    return ClaimPredictor(
        embedder=embedder, model_path=MODEL_PATH, feature_spec_path=FEATURE_SPEC_PATH
    )


@lru_cache
def get_compiled_graph():
    secrets = get_secrets()
    langfuse = get_langfuse()
    predictor = get_predictor()
    anthropic_client = Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])
    return build_graph(predictor, anthropic_client, langfuse)
