# FastAPI Python — Code Patterns & Best Practices

This skill describes project-agnostic patterns for writing clean, maintainable FastAPI services in Python. Follow these guidelines when generating or reviewing code.

---

## Project Layout

```
app/
  main.py          # FastAPI app, routes, Pydantic schemas — nothing else
utils/
  config.py        # Static constants and feature flags
  secrets_utils.py # Environment variable loading
  app_utils.py     # Core service and domain classes
  <domain>_utils.py # One file per bounded concern (db, client, cache, etc.)
```

**Rules:**
- `main.py` is the HTTP boundary only — it defines routes and schemas, delegates everything to service classes.
- Each `utils/` file owns one concern. Never mix unrelated concerns (e.g. DB logic and external-API logic) in the same file.
- `config.py` holds values that change between environments but are not secrets (timeouts, page sizes, feature flags). No logic.
- `secrets_utils.py` reads `os.getenv()` in one place and returns a typed dict. No secrets scattered across files.

---

## Pydantic Schemas

Define all request and response schemas as Pydantic `BaseModel` classes at the top of `main.py`, before routes.

```python
from pydantic import BaseModel, Field

class CreateItemRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: str = "default"

class ItemResponse(BaseModel):
    id: str
    name: str
    createdAt: str
```

**Rules:**
- Every route has an explicit `response_model`. Never return raw dicts without a schema.
- Use `Field(...)` to add validation constraints at the boundary. Don't validate the same thing again inside service code.
- Request models live in `main.py`. Complex domain objects (not HTTP schemas) live in `utils/`.

---

## Thin Routes

Routes delegate immediately to service classes. No business logic in route functions.

```python
@app.post("/api/v1/items", response_model=ItemResponse)
async def create_item(request: CreateItemRequest):
    item = service.create(request.name, request.category)
    return {"id": item.id, "name": item.name, "createdAt": item.created_at}
```

**Rules:**
- Route body: one service call + return. If it's longer, extract to a service method.
- Only raise `HTTPException` in routes (or in service methods that are inherently HTTP-aware, like session lookup).
- Use `logger.error()` before re-raising unexpected exceptions — don't swallow them silently.

---

## Service Classes (Single Responsibility)

Each class owns one domain concept. Instantiation wires dependencies; methods do work.

```python
class App:
    def __init__(self):
        self.secrets = get_secrets()
        self.db = get_db_manager(self.secrets)
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create_session(self, user_id: str) -> "Session":
        session = Session(user_id, self)
        with self._lock:
            self.sessions[user_id] = session
        return session

    def get_session(self, session_id: str) -> "Session":
        with self._lock:
            session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return session
```

**Rules:**
- **S** — One class, one responsibility. `App` manages sessions; `Session` manages per-session state; `DatabaseManager` manages DB I/O.
- **D** — Pass dependencies (config, secrets, client instances) into constructors. Never `import` a singleton from inside a method.
- **O** — Extend behavior through parameters or composition, not by modifying existing classes.
- Private implementation methods get an `_underscore` prefix. Public methods are the interface.

---

## Singleton Services with `lru_cache`

Use `@lru_cache()` on a factory function to create a process-wide singleton without a global variable.

```python
from functools import lru_cache

@lru_cache()
def get_app() -> App:
    return App()
```

Call `get_app()` wherever you need the instance — in routes, in tests, anywhere. The cache ensures one instance per process.

**Rules:**
- Only cache stateless-at-construction objects (the object may hold mutable state internally, but the factory function takes no mutable arguments).
- Don't call `get_app()` at module import time. Call it inside route functions or in a `lifespan` handler so startup errors surface cleanly.

---

## External Client Wrappers

Wrap third-party clients in a class. Private methods handle protocol details; public methods expose domain operations.

```python
class ApiClient:
    def __init__(self, base_url: str, token: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.base_url = base_url

    def _get(self, path: str, params: dict = None) -> dict:
        for attempt in range(3):
            r = self.session.get(f"{self.base_url}{path}", params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {path} failed after 3 attempts")

    def _paginate(self, path: str, params: dict = None) -> Iterator[dict]:
        params = dict(params or {})
        page = 1
        while True:
            params["page"] = page
            resp = self._get(path, params=params)
            yield from resp.get("data", [])
            if page >= resp.get("meta", {}).get("pagination", {}).get("total_pages", 1):
                break
            page += 1

    def list_items(self, category: str = None) -> List[dict]:
        params = {"category": category} if category else {}
        return list(self._paginate("/v3/items", params=params))
```

**Rules:**
- `_get` / `_paginate` / `_call` are private implementation details. Callers use `list_items()`.
- Retry and rate-limit logic lives inside the wrapper, not in callers.
- Inject `base_url` and `token` at construction; never read `os.getenv` inside this class.

---

## Database Access

Use a context manager to encapsulate connection lifecycle. Never open connections in route handlers.

```python
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_config: dict, max_conn: int = 10):
        self.pool = pool.SimpleConnectionPool(1, max_conn, **db_config)

    @contextmanager
    def get_connection(self):
        conn = self.pool.getconn()
        try:
            cur = conn.cursor()
            yield conn, cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            self.pool.putconn(conn)

    def insert_record(self, data: dict):
        with self.get_connection() as (conn, cur):
            cur.execute("INSERT INTO items (name) VALUES (%s)", (data["name"],))
```

**Rules:**
- All SQL lives in `DatabaseManager` methods — never inline SQL in service classes or routes.
- Use parameterized queries (`%s` placeholders). Never interpolate user input into SQL strings.
- Rollback on any exception; always return the connection to the pool in `finally`.

---

## Configuration and Secrets

```python
# utils/config.py — static, committed to git
CONFIG = {
    "request_timeout": 30,
    "page_size": 25,
    "max_retries": 3,
}

# utils/secrets_utils.py — reads env, never committed
def get_secrets() -> dict:
    return {
        "DB_CONFIG": {
            "host": os.getenv("DB_HOST"),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        },
        "API_KEY": os.getenv("API_KEY"),
    }
```

**Rules:**
- `config.py` contains no secrets and no `os.getenv` calls.
- `secrets_utils.py` contains no business logic — only reads env vars and packages them.
- Call `get_secrets()` once at `App.__init__`. Pass the result down; don't re-read env vars in nested classes.

---

## Thread Safety

Shared mutable state accessed by concurrent requests must be protected by a lock.

```python
class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def add(self, session_id: str, session: Session):
        with self._lock:
            self._sessions[session_id] = session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)
```

**Rules:**
- Lock only the dict access — not the entire business operation — to minimise contention.
- Read and write always use the lock consistently. A read without a lock is a race condition.

---

## Utility Functions

Stand-alone helpers that don't belong to a class go in the relevant `utils/` file as module-level functions.

```python
# utils/app_utils.py
def timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
```

**Rules:**
- **DRY** — extract repeated expressions into a named function once, even if it's one line, if it has semantic meaning.
- **YAGNI** — don't add parameters or abstractions for hypothetical future callers. If the function is only ever called one way, keep it simple.
- **KISS** — if a utility function needs more than ~15 lines, reconsider: it likely belongs in a class or should be split.

---

## Error Handling

```python
@app.post("/api/v1/items/{item_id}/process", response_model=ProcessResponse)
async def process_item(item_id: str, request: ProcessRequest):
    item = service.get_item(item_id)  # raises 404 HTTPException if missing
    try:
        result = service.process(item, request.options)
        return {"itemId": item_id, "result": result}
    except Exception as e:
        logger.error(f"Process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Rules:**
- Domain `HTTPException` (404, 400, 422) is raised in service methods close to the validation point.
- Unexpected exceptions are caught only at the route level, logged, then re-raised as 500.
- Never catch `Exception` silently — always log before re-raising or transforming.
- Don't add `try/except` for code paths that cannot raise. Validate at the boundary (input), trust internal code.

---

## Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Classes | `PascalCase` | `DatabaseManager`, `SessionStore` |
| Public methods | `snake_case` | `get_session`, `list_items` |
| Private methods | `_snake_case` | `_call`, `_parse_json` |
| Pydantic models | `PascalCase` + suffix | `CreateItemRequest`, `ItemResponse` |
| Factory functions | `get_<thing>` | `get_app`, `get_db_manager` |
| Constants | `UPPER_SNAKE_CASE` | `CONFIG`, `DEFAULT_TIMEOUT` |
| Private constants | `_UPPER_SNAKE_CASE` | `_INTERNAL_DEFAULTS` |

---

## Checklist for New Features

- [ ] Route body is ≤ 5 lines; business logic is in a service class
- [ ] Every route has a `response_model`
- [ ] No `os.getenv` outside `secrets_utils.py`
- [ ] No SQL outside `DatabaseManager`
- [ ] External client wrapped in a class with private protocol methods
- [ ] Shared mutable state protected by a lock
- [ ] New utility function placed in the correct `utils/` file, not inlined
- [ ] No `except Exception: pass` — always log