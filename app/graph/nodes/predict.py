from app.graph.state import ClaimState
from app.model.predictor import ClaimPredictor


class PredictNode:
    def __init__(self, predictor: ClaimPredictor):
        self._predictor = predictor

    def __call__(self, state: ClaimState) -> dict:
        row = self._predictor.build_feature_row(state["claim"])
        decline_probability, decision = self._predictor.predict(row)
        return {
            "feature_row": row,
            "decline_probability": decline_probability,
            "decision": decision,
        }
