from anthropic import Anthropic
from langfuse import Langfuse

from app.config import DEFAULT_LANGUAGE, LANGUAGES
from app.graph.state import ClaimState
from app.model.predictor import ClaimPredictor

TOP_FACTOR_COUNT = 6


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


class AttributeNode:
    def __init__(self, predictor: ClaimPredictor):
        self._predictor = predictor

    def __call__(self, state: ClaimState) -> dict:
        tabular_shap, text_contributions = self._predictor.shap_values(state["feature_row"])
        return {
            "tabular_shap": tabular_shap,
            "text_contributions": text_contributions,
        }


class ExplainNode:
    def __init__(
        self,
        persona: str,
        prompt_template: str,
        client: Anthropic,
        model: str,
        max_tokens: int,
        langfuse: Langfuse | None = None,
    ):
        self._persona = persona
        self._prompt_template = prompt_template
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._langfuse = langfuse

    def __call__(self, state: ClaimState) -> dict:
        prompt = self._build_prompt(state)
        messages = [{"role": "user", "content": prompt}]
        explanation = self._call_llm(messages)
        return {"persona_explanations": {self._persona: explanation}}

    def _call_llm(self, messages: list[dict]) -> str:
        if self._langfuse is None:
            response = self._client.messages.create(
                model=self._model, max_tokens=self._max_tokens, messages=messages
            )
            return response.content[0].text
        with self._langfuse.start_as_current_observation(
            name=f"explain-{self._persona}", as_type="generation", model=self._model, input=messages
        ) as generation:
            response = self._client.messages.create(
                model=self._model, max_tokens=self._max_tokens, messages=messages
            )
            explanation = response.content[0].text
            generation.update(
                output=explanation,
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            return explanation

    def _build_prompt(self, state: ClaimState) -> str:
        top_factors = sorted(
            state["tabular_shap"].items(), key=lambda item: abs(item[1]), reverse=True
        )[:TOP_FACTOR_COUNT]
        factors_text = "\n".join(f"- {name}: {value:+.4f}" for name, value in top_factors)

        text_contributions = state["text_contributions"]
        language_code = state["claim"].get("language") or DEFAULT_LANGUAGE
        return self._prompt_template.format(
            decision=state["decision"],
            decline_probability=f"{state['decline_probability']:.2%}",
            contributing_factors=factors_text,
            issuedesc_contribution=f"{text_contributions['issueDesc']:+.4f}",
            other_contribution=f"{text_contributions['other']:+.4f}",
            issue_description=state["claim"].get("issueDesc") or "(none provided)",
            other_description=state["claim"].get("other") or "(none provided)",
            response_language=LANGUAGES.get(language_code, LANGUAGES[DEFAULT_LANGUAGE]),
        )


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
            "text_contributions": state["text_contributions"],
            "explanations": explanations,
        }
        return {"response": response}
