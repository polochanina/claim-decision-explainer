from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class ClaimState(TypedDict, total=False):
    claim: dict[str, Any]
    feature_row: Any
    decline_probability: float
    decision: str
    tabular_shap: dict[str, float]
    text_contribution: float
    persona_explanations: Annotated[dict[str, str], merge_dicts]
    response: dict[str, Any]
