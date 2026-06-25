# Python Code Patterns

Conventions for writing Python code in any project in this style.
Follow these when generating or modifying Python code.

---

## Encapsulation: behavior lives in classes

Every logical unit of behavior is a class. Only two things live at module level:
- **Pure constants** (dicts, strings, numbers)
- **The entry-point function** (the one thing that wires everything together)

Dependencies are injected via `__init__`, not fetched inside methods.

```python
# correct
class ApiClient:
    def __init__(self, model: str, api_key: str):
        self.client = SomeSDK(api_key=api_key)
        self.model = model

# wrong — fetches its own dependencies
class ApiClient:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
```

---

## File and folder structure

```
main.py / app.py       # Entry point only — wires things together, no logic
utils/
  config.py            # App-level constants
  secrets_utils.py     # Reads env vars, returns a dict
  <domain>.py          # One class per file, one responsibility per file
```

**One responsibility per file.** If a file grows two unrelated concerns, split it.
New utilities go in `utils/`. New entry points go at the project root.

---

## SOLID

### Single Responsibility Principle
Each class answers one question: "what do I do?"

- An API client talks to the API — it does not render UI.
- A UI class renders output — it does not call external services directly.
- A config file holds constants — it does not read from disk or environment.

### Open/Closed Principle
Extend by adding to data structures, not by editing callers.

```python
# correct — add behavior by adding a key; callers are unchanged
def get_prompts() -> dict:
    return {
        "validate": VALIDATE_PROMPT,
        "adjust": ADJUST_PROMPT,
        "summarise": SUMMARISE_PROMPT,   # new, no caller changes needed
    }

# wrong — callers must be updated each time
def get_validate_prompt(): ...
def get_adjust_prompt(): ...
```

### Dependency Inversion Principle
High-level classes receive dependencies; they do not create or fetch them.

```python
# correct — dependencies are passed in
class Processor:
    def __init__(self, client: ApiClient, config: dict): ...

# wrong — Processor decides where to get things
class Processor:
    def __init__(self):
        self.client = ApiClient(api_key=os.getenv("API_KEY"))
        self.config = load_config("config.json")
```

**Exception — top-level service/pipeline classes and composition-root factories.**
A class or factory function that sits at the *root* of the dependency graph
(one per process, instantiated once by the entry point — e.g. `IngestionPipeline`,
or `llm_utils.engine_factory.build_rag_engine`) is allowed to build its own
adapters via the single shared secrets loader:

```python
# correct — root-level service builds its own adapters from secrets
class IngestionPipeline:
    def __init__(self, mit: bool, wikipedia: bool, ingestion_config: dict) -> None:
        secrets = get_secrets()
        self._store = QdrantVectorStore(url=secrets["QDRANT_URL"], ...)
        self._embedder = VoyageEmbedder(api_key=secrets["VOYAGE_API_KEY"])
```

This keeps the entry point a one-line wire-up (`IngestionPipeline(...).run()`)
instead of a long block of adapter construction the caller has no use for.
Anything *not* at the root — collaborators, helpers, anything constructed more
than once — still receives its dependencies via `__init__` per the rule above.

---

## DRY — Don't Repeat Yourself

Extract any logic used more than once into a private method or shared constant.

```python
# correct — one helper, called with different arguments
self._render_counter(count_a, max_a, selector_a)
self._render_counter(count_b, max_b, selector_b)

# wrong — the same block copy-pasted twice
```

All user-visible strings live in a dedicated constants file.
A string that appears in more than one place must be a named constant, never a repeated literal.

---

## KISS — Keep It Simple

- Use plain `dict` for static data — not a class with getters.
- Keep entry-point functions short. Do not add logic to them.
- Do not introduce base classes or inheritance until two concrete classes share significant, non-trivial behavior.
- No `argparse` (or other CLI-parsing) for internal/dev-run scripts. State the choice
  directly as explicit function parameters or a constant the next person edits in place.
- A method that only forwards to `asyncio.gather(self.a(), self.b())` with no
  added behavior is a redundant wrapper — let the caller `gather` directly.

```python
# correct
CONFIG = {"name": "MyApp", "version": "1.0"}

# wrong — unnecessary class for static data
class Config:
    name = "MyApp"
    version = "1.0"
```

```python
# correct — explicit, edited directly by whoever runs it
async def run_ingestion(mit: bool, wikipedia: bool) -> None: ...
asyncio.run(run_ingestion(mit=True, wikipedia=True))

# wrong — argparse ceremony for a single-developer internal script
parser.add_argument("--sources", nargs="+", choices=["mit", "wikipedia"])
```

**Group related tunables into one config dict, passed whole — not unpacked.**
When a constructor would otherwise collect five-plus loosely related `int`/`str`
defaults, move them into a single named dict at the call site and accept that
dict as one parameter. Don't spread it back out with `**config` — that just
re-creates the long parameter list on the receiving end.

```python
# correct
ingestion_config = {"play_token_cap": 600, "wiki_max_articles": 5000, ...}
IngestionPipeline(mit=True, wikipedia=True, ingestion_config=ingestion_config)

def __init__(self, mit: bool, wikipedia: bool, ingestion_config: dict) -> None:
    self._config = ingestion_config   # accessed as self._config["play_token_cap"]

# wrong — unpacking just rebuilds the long parameter list
IngestionPipeline(mit=True, wikipedia=True, **ingestion_config)
```

---

## YAGNI — You Aren't Gonna Need It

Do not add the following unless a specific, present need exists:

- Logging frameworks (use the framework's native error display instead)
- Retry logic (add only when flakiness is observed and measured)
- Abstract base classes (add only when multiple concrete implementations exist)
- Caching layers (add only when a performance problem is measured)
- Persistence / databases (add only when stateless processing is insufficient)

---

## Imports and literals

- Imports live at the top of the file with the rest of the module's imports.
  Only import inside a function/method to break a genuine circular import —
  not as a default habit.
- Plain integer literals do not use `_` digit-group separators: `5000`, not `5_000`.

---

## Comments

Do not use dash-bordered banner comments to mark out sections of a class or
module (`# ----...`). The class/file structure and method names already convey
grouping; a banner is one more thing to maintain and it adds visual noise without
adding information.

```python
# correct — methods grouped by blank lines and names alone
class IngestionPipeline:
    async def run(self) -> None: ...

    async def _run_mit(self) -> None: ...

    async def _run_wikipedia(self) -> None: ...

# wrong — banner comments restate what the names already say
class IngestionPipeline:
    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(self) -> None: ...
```

---

## Method visibility

| Prefix | Meaning |
|--------|---------|
| no prefix | Public API — may be called by code outside the class |
| `_` single underscore | Private helper — called only within the class |

Public methods form the contract. Private methods are implementation detail.

---

## Type hints

All method signatures carry type hints — parameters and return types.

```python
from typing import Tuple, List, Dict, Optional

def validate(self, instruction: str) -> Tuple[bool, str]: ...
def process(self, items: List[Dict[str, str]], max_count: int) -> Optional[str]: ...
```

Do not add `from __future__ import annotations`. This project targets Python
3.11+, where built-in generics (`list[str]`, `dict[str, int]`) and `X | None`
unions already work natively at runtime — the future import exists to backport
that syntax to older interpreters, which is not a concern here. Add it only if
a file genuinely needs lazy annotation evaluation (e.g. a class method whose
signature references its own not-yet-defined class by name, unquoted).

---

## Error handling

Catch exceptions only at system boundaries — external API calls and entry points.
Do not wrap internal, pure logic in `try/except`.

```python
# correct — catch at the external boundary
def call_api(self, prompt: str) -> str:
    try:
        return self.client.send(prompt)
    except Exception as e:
        return f"Error: {e}"

# wrong — catching inside a pure internal helper
def _build_prompt(self, text: str) -> str:
    try:
        return f"Text: {text}"
    except Exception:
        return ""
```

The top-level `run()` method (or equivalent) should have one outer `try/except`
to surface unexpected errors gracefully rather than crashing silently.
