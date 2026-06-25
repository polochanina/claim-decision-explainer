from fastapi import FastAPI

from app.config import EXPLAIN_CLAIM_TRACE_NAME
from app.graph.build import get_compiled_graph, get_predictor
from app.observability import get_langfuse
from app.schemas import ClaimRequest, ExplanationResponse

app = FastAPI(title="Claim Approval Agent")


@app.get("/health")
async def health() -> dict:
    predictor = get_predictor()
    return {"status": "ok", "model_lineage": predictor.model_lineage}


@app.post("/explain-claim", response_model=ExplanationResponse)
async def explain_claim(request: ClaimRequest) -> dict:
    graph = get_compiled_graph()
    claim = request.model_dump()
    langfuse = get_langfuse()

    if langfuse is None:
        result = graph.invoke({"claim": claim})
        return result["response"]

    with langfuse.start_as_current_observation(
        name=EXPLAIN_CLAIM_TRACE_NAME,
        as_type="span",
        input={"productName": claim.get("productName"), "claimType": claim.get("claimType")},
    ) as span:
        result = graph.invoke({"claim": claim})
        response = result["response"]
        span.update(
            output={
                "decision": response["decision"],
                "approval_probability": response["approval_probability"],
            }
        )
        return response
