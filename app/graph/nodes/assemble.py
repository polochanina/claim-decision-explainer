from app.graph.state import ClaimState


class AssembleNode:
    def __call__(self, state: ClaimState) -> dict:
        contributing_factors = [
            {"feature": name, "contribution": value}
            for name, value in sorted(
                state["tabular_shap"].items(), key=lambda item: abs(item[1]), reverse=True
            )
        ]
        explanations = [
            {"persona": persona, "explanation": text}
            for persona, text in state["persona_explanations"].items()
        ]

        response = {
            "decision": state["decision"],
            "approval_probability": 1 - state["decline_probability"],
            "contributing_factors": contributing_factors,
            "text_contribution": state["text_contribution"],
            "explanations": explanations,
        }
        return {"response": response}
