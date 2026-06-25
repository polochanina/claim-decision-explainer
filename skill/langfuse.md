# Langfuse Observability Patterns

Conventions for instrumenting LLM calls with Langfuse in this style.
Follow these when adding tracing/observability to `RAGEngine` or any other
class that talks to an LLM.

This project pins `langfuse>=4` — the OpenTelemetry-based SDK. Older patterns
(`langfuse.trace(...)`, `langfuse_context`, `langfuse.anthropic`) belong to the
v2 SDK and **do not exist** on this version; using them raises
`AttributeError: 'Langfuse' object has no attribute 'trace'`.

---

## Setup

```bash
pip install "langfuse>=4"
```

Add keys to `.env` and read them through `secrets_utils.py`, the same way
`ANTHROPIC_KEY` is read — never call `os.getenv` from inside a service class.

```python
# utils/secrets_utils.py
def get_secrets():
    secrets = {
        "ANTHROPIC_KEY": os.getenv("ANTHROPIC_KEY").strip('"'),
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    }
    return secrets
```

---

## The tracer is a dependency, not something the engine fetches

Following the project's DIP convention, `RAGEngine` receives a configured
`Langfuse` instance (or `None`) through `__init__` — it does not construct its
own, and tracing must be optional so the engine works without Langfuse keys.

```python
# correct
class RAGEngine:
    def __init__(self, vector_store, embedder, llm_client, langfuse: Langfuse | None = None):
        self._langfuse = langfuse

# wrong — class reaches into the environment itself
class RAGEngine:
    def __init__(self, ...):
        self._langfuse = Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), ...)
```

Build the `Langfuse` instance once in `engine_factory.py` from the `secrets`
dict, alongside `QdrantVectorStore` and the `Anthropic` client.

---

## Create a trace by opening a root span

In v4 there is no standalone `trace` object — a trace is simply the root
`span` of an OpenTelemetry trace tree. Use `start_as_current_observation` as a
context manager, name it after the user-facing operation (this project has
exactly one: `_TRACE_NAME = "shakespeare-rag"`), and call `.update(...)` on the
returned span to attach output once the work completes.

```python
# correct
with self._langfuse.start_as_current_observation(
    name=_TRACE_NAME, as_type="span", input={"question": question}
) as span:
    try:
        answer = await self._run(question, chat_history)
        span.update(output={"answer": answer.text, "citations": len(answer.cited_sources)})
        return answer
    except Exception as e:
        span.update(output={"error": str(e)})
        raise

# wrong — v2 API, no longer exists on Langfuse in v4
trace = self._langfuse.trace(name=_TRACE_NAME, input={"question": question})
trace.update(output=...)
```

Nested LLM calls within the `with` block are automatically attached as child
observations as long as they run through an instrumented client — no manual
span wiring needed for the children.

---

## Capture model + token usage with explicit `generation` spans

There is no `langfuse.anthropic` wrapper in v4 (it was a v2-only convenience),
and no Anthropic OTel auto-instrumentation is installed in this project. A
plain `anthropic.AsyncAnthropic` call made inside a `with
start_as_current_observation` block nests correctly as a child span, but
without `model`/`usage_details` Langfuse cannot compute token counts or cost —
the dashboard shows the call with no cost figure.

So every `self._llm.messages.create` call goes through `RAGEngine._call_llm`,
which opens an `as_type="generation"` span, and reports `model` plus
`usage_details` mapped from the Anthropic response's `usage.input_tokens` /
`usage.output_tokens`:

```python
# correct — model + usage reported, Langfuse can compute cost
async def _call_llm(self, name: str, **kwargs: Any) -> Any:
    if self._langfuse is None:
        return await self._llm.messages.create(**kwargs)
    with self._langfuse.start_as_current_observation(
        name=name, as_type="generation", model=kwargs.get("model"), input=kwargs.get("messages")
    ) as generation:
        response = await self._llm.messages.create(**kwargs)
        generation.update(
            output=response.content[0].text,
            usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )
        return response

# wrong — bare call inside the root span has no model/usage, no cost in the dashboard
response = await self._llm.messages.create(model=self._model, messages=messages)
```

If an Anthropic OTel instrumentation package is added later, prefer it and
delete `_call_llm` rather than running both — don't double-report usage.

---

## Tie traces to the Streamlit session, not to a random id

Langfuse traces should carry `session_id` and, if available, `user_id` so
that a full user journey groups together in the dashboard. Pass them via
`trace_context` (or as `metadata`) on `start_as_current_observation`, reusing
the id already stored in `st.session_state` rather than generating a new one
per request.

```python
# correct — reuse the id already stored in session
self.session.setdefault("trace_session_id", str(uuid.uuid4()))
...
with self._langfuse.start_as_current_observation(
    name=_TRACE_NAME,
    as_type="span",
    input={"question": question},
    metadata={"session_id": session_id},
) as span:
    ...

# wrong — a fresh id every rerun, fragmenting the trace
metadata={"session_id": str(uuid.uuid4())}
```

UI concerns (reading `st.session_state`) stay in `streamlit_ui/`; `RAGEngine`
accepts `session_id`/`user_id` as plain parameters.

---

## Don't let tracing break the app

Tracing is observability, not a system boundary the user depends on. If
Langfuse is unreachable, the app must still answer the user. `RAGEngine`
already guards this by accepting `langfuse: Langfuse | None` and skipping
tracing entirely when it's `None` — keep that early-exit at the top of
`answer_question`. Flush/shutdown failures should be swallowed where the
client is torn down (e.g. `engine_factory` or app lifespan), the same way
unexpected exceptions are caught there.

```python
# correct — tracing absence/failure never blocks the answer
async def answer_question(self, question, chat_history):
    if self._langfuse is None:
        return await self._run(question, chat_history)
    with self._langfuse.start_as_current_observation(...) as span:
        ...
```

---

## No hardcoded strings, no secrets in code

Trace/observation names are short identifiers, not user-facing copy, so they
stay as plain string constants at the call site (`_TRACE_NAME = "shakespeare-rag"`)
— they do not belong in `ui_text.py`. Public/secret keys and the host URL
always come from `secrets_utils.get_secrets()`, never hardcoded or committed.

---

## YAGNI

Do not add the following unless a specific, present need exists:
- Custom scoring / evaluation pipelines (`run_batched_evaluation`,
  `run_experiment`) — add only when there is a defined quality metric to track
- Prompt management via Langfuse (this project manages prompts as plain string
  constants in `engine.py` — don't duplicate that source of truth)
- The `@observe` decorator — this project instruments at the `answer_question`
  boundary with an explicit `start_as_current_observation` span; mixing both
  styles creates two trace-naming conventions for one pipeline
- Self-hosted Langfuse (use Langfuse Cloud via `LANGFUSE_HOST` until there is
  a concrete reason — compliance, volume — to self-host)
