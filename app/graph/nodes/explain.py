from anthropic import Anthropic
from langfuse import Langfuse

from app.graph.state import ClaimState

TOP_FACTOR_COUNT = 6


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

        return self._prompt_template.format(
            decision=state["decision"],
            decline_probability=f"{state['decline_probability']:.2%}",
            contributing_factors=factors_text,
            text_contribution=f"{state['text_contribution']:+.4f}",
            issue_description=state["claim"].get("issueDesc") or "(none provided)",
        )
