from app.graph.state import ClaimState
from app.model.predictor import ClaimPredictor


class AttributeNode:
    def __init__(self, predictor: ClaimPredictor):
        self._predictor = predictor

    def __call__(self, state: ClaimState) -> dict:
        tabular_shap, text_contribution = self._predictor.shap_values(state["feature_row"])
        return {
            "tabular_shap": tabular_shap,
            "text_contribution": text_contribution,
        }
