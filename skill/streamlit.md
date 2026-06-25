# Streamlit Patterns

Conventions for writing Streamlit applications in this style.
Follow these when generating or modifying any Streamlit UI code.

---

## Entry point is a thin wire-up

The entry-point file (`main.py` or `app.py`) does three things only:
1. Call `st.set_page_config` (must run before any other `st.*` call)
2. Initialise session state once
3. Delegate to `ui.run()`

No rendering, no business logic.

```python
def run_app():
    st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")
    if "initialised" not in st.session_state:
        st.session_state["config"] = CONFIG
        st.session_state["ui"] = AppUI(st.session_state)
        st.session_state["initialised"] = True
    st.session_state["ui"].run()
```

The `if "key" not in st.session_state` guard is essential: Streamlit reruns the
entire script on every widget interaction, so without the guard you recreate objects
and reset state on every click.

---

## Always launch via the Streamlit CLI, never `python entry_point.py`

The entry-point file must be started with `streamlit run <file>` (or
`python -m streamlit run <file>`). Running it as a plain script
(`python run_streamlit.py`, an IDE "Run" button that shells out to `python`, a
debugger launch config, etc.) executes `st.*` calls with no `ScriptRunContext`,
which produces noisy `missing ScriptRunContext` / `Session state does not
function` warnings and a non-functional app — `st.session_state` silently no-ops,
so the `if "initialised" not in st.session_state` guard never holds and nothing
renders interactively.

These warnings are a launch-method symptom, not a code defect — don't try to
silence them with logging config or `warnings.filterwarnings`. Fix the run
command/IDE config instead:

```powershell
streamlit run run_streamlit.py
```

---

## All UI code lives in one class

Never write `st.*` calls outside the UI class.
The entry-point file is the only exception (`st.set_page_config`).

```python
# correct
class AppUI:
    def display_header(self):
        st.title(self.ui_text["title"])

# wrong — st calls at module level
st.title("My App")
```

---

## Method structure

| Method type | Naming | Role |
|-------------|--------|------|
| Public | `display_*` | Renders a major UI section (header, sidebar, results) |
| Public | `show_*` | Renders a transient state (error, warning, success) |
| Private | `_render_*` | Renders a reusable sub-component |
| Private | `_handle_*` | Processes a user action (form submit, button click) |
| Public | `run()` | Orchestrates the full render cycle with error handling |

`run()` is the only method called from outside the class.

---

## Session state is the single source of truth

All mutable state lives in `self.session` (a reference to `st.session_state`).
Methods write to it; they do not return data to each other.

```python
# correct — write to session in one method, read in another
def _handle_submit(self, text: str):
    self.session["result"] = self._process(text)

def _display_output(self):
    if self.session.get("result"):
        st.markdown(self.session["result"])

# wrong — trying to pass data between render methods via return values
def _handle_submit(self, text: str) -> str:
    return self._process(text)
```

Always clear related state at the start of a new submission to avoid stale output:

```python
self.session["result"] = None
self.session["error"] = None
```

---

## No hardcoded strings in the UI class

Every user-visible string comes from a dedicated `ui_text` dict (or equivalent constant).

```python
# correct
st.warning(self.ui_text["warn_empty_input"])

# wrong
st.warning("Please enter some text first.")
```

When adding new UI copy, add the key to all supported languages in the constants
file before using it in the class.

---

## Language / locale switching

When switching locale:
1. Update the lang key in session
2. Reset all output and input-derived state to `None`
3. Remove widget keys so Streamlit re-renders them fresh
4. Call `st.rerun()`

```python
if selected_lang != self.session["lang"]:
    self.session["lang"] = selected_lang
    self.session["result"] = None
    for key in ["input_text", "preset_choice"]:
        self.session.pop(key, None)
    st.rerun()
```

Never call `st.rerun()` outside of an explicit user action.

---

## Sidebar for settings, main area for content

User-configurable parameters (limits, toggles, options) belong in `st.sidebar`.
The main column contains the primary task flow: input → action → output.

---

## Spinners wrap slow operations

Wrap every call that may take more than ~0.5 s in `st.spinner`.
The label comes from `ui_text`, never hardcoded.

```python
with st.spinner(self.ui_text["spinner_loading"]):
    result = self.service.process(input_text)
```

---

## Custom HTML/JS components

Use `st.components.v1.components.html()` only for behaviour Streamlit cannot
provide natively (e.g., live counters that update without a full rerun).

Keep the HTML/JS inside a dedicated `_render_*` method — never inline in a
`display_*` method — so it is reusable and the calling code stays readable.

Pass Python values into the script via f-string interpolation.
Always set an explicit `height=` to prevent unwanted scrollbars.

---

## Error handling in run()

`run()` wraps the entire render cycle so unexpected exceptions surface as a
readable error rather than a blank or broken page:

```python
def run(self):
    try:
        self.display_header()
        self.display_main()
    except Exception as e:
        self.show_error(e)

def show_error(self, e: Exception):
    st.error(self.ui_text["error_title"])
    with st.expander("Details"):
        st.code(traceback.format_exc())
```
