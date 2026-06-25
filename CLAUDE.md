# CLAUDE.md

Guidance for working in this repository: a FastAPI service that wraps a pre-trained
CatBoost claim-approval model in a LangGraph pipeline, producing multi-persona GenAI
explanations for each decision.

## What's already done vs. what this app does

`notebooks/EDA + modeling.ipynb` is the offline training notebook — already run to completion.
It exports three artifacts this app loads at startup and never regenerates:

- `artifacts/catboost_model.cbm` — trained CatBoost model (tabular + text-embedding features).
- `artifacts/feature_spec.json` — the train/serve contract (see below).
- `artifacts/sample_claim.json` — one real raw claim, used as the canonical example request.

Do not retrain or re-export from the notebook as part of app work. If the notebook changes
and re-exports, the app's `predictor.py` must be re-validated against the new
`feature_spec.json`, not edited to silently tolerate drift.

## Train/serve contract — read this before touching `model/predictor.py`

Column order and preprocessing must match training **exactly**, or the model silently
produces garbage predictions (CatBoost aligns features by position, not by name lookup
at inference unless you pass a DataFrame with matching columns — always pass columns in
`feature_spec.json["feature_order"]` order).

- **Tabular columns** (`feature_spec.json["tabular_columns"]`): `excessFee, rrp,
  productName, productDesc, policyStatus, retailerName, deviceType, model, channel,
  claimType, country, turnOnOff, touchScreen, frontCamera, backCamera, audio, mic,
  buttons, connection, charging`.
- **Categorical** (`feature_spec.json["cat_features"]`, subset of the above): `productName,
  productDesc, policyStatus, retailerName, deviceType, model, channel, claimType,
  country`. Missing values become the literal string `"missing"`, then all values are
  cast to `str`. The remaining tabular columns are numeric/boolean form-checkbox flags —
  leave missing values as `NaN`; CatBoost handles numeric NaN natively. Do not fillna them.
- **`other`** is dropped entirely — never a model input, in training or serving.
- **`issueDesc`** is never a raw tabular feature. It is embedded live via Voyage
  (`model="voyage-4"`, `input_type="document"`, `output_dimension=256`) into columns
  `emb_issueDesc_0..255`, appended after the tabular columns. An empty/missing `issueDesc`
  is embedded as a single space `" "` (not skipped, not zero-filled) — matches training.
- **Target polarity**: `target == 1` means **declined**. `predict_proba(...)[:, 1]` is
  P(decline). Any "approval probability" surfaced in the API or to an LLM prompt is
  `1 - P(decline)` — make this conversion once, in `predictor.py`, and never re-derive it
  downstream.

If you ever need to change this contract, it must originate from a notebook re-export of
`feature_spec.json`, not a hand-edit in the app.

## Secrets

`ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` are read from `.env` via `app/secrets_utils.py`
only. No other file calls `os.getenv`. Never hardcode a key in source, a notebook cell, or
a test fixture — `notebooks/EDA + modeling.ipynb` cell 39 currently contains a hardcoded Voyage key
left over from notebook experimentation; treat that as a leaked credential to be rotated,
not a pattern to copy.

## Code conventions

This project follows the two skill files already in `skill/fastapi.md` and
`skill/python.md`. The short version:

- **Behavior lives in classes**; only pure constants and the entry-point function live at
  module level. Dependencies are passed into `__init__`, not fetched inside methods.
- `app/secrets_utils.py` is the only `os.getenv` call site, returning a typed dict.
  `app/config.py` holds non-secret constants (persona list, Claude model id, file paths) —
  no logic, no env reads.
- `app/main.py` is the HTTP boundary only: routes + `response_model`, one service/graph
  call per route, nothing else. Business logic lives in `app/graph/` and `app/model/`.
- External clients (`VoyageEmbedder`, the Anthropic client, `ClaimPredictor`) are wrapped in
  classes with private protocol methods and a small public domain-method surface
  (`embed`, `predict`, `explain`).
- LangGraph nodes are thin classes, one per file under `app/graph/nodes/`, each constructed
  once via an `@lru_cache` factory in `app/graph/build.py` and given its dependencies
  (predictor, embedder, Claude client) at construction — not fetched inside `__call__`.
- Type hints on every signature. No `from __future__ import annotations` (Python 3.11+,
  native generics work).
- No banner comments (`# ---- Section ----`). Group by blank lines and names.
- Catch exceptions only at boundaries (the FastAPI route, the external API call inside a
  client wrapper). Don't wrap pure internal logic in `try/except`.
- No `argparse` for internal scripts — explicit parameters or constants instead.

## Prompt design constraint (binding, not stylistic)

Both persona prompts in `app/prompts/` instruct Claude to **explain the model's decision,
never second-guess it**. If `issueDesc` reads as deniable but the model approved (or vice
versa), the adjuster explanation must surface that tension as an explicit review flag —
it must not imply the decision should be different. This is what keeps explanations
faithful to the SHAP-grounded evidence rather than an independent LLM judgment call.

## Out of scope for this app (do not build)

- Auth, rate-limiting, streaming responses — note as production concerns in `docs/DESIGN.md`,
  don't implement.
- Embedding caching at serve time — every request embeds `issueDesc` live; that's correct
  for novel claims, not a gap to fix.
- Synthetic claim scenario generation — an offline notebook concern, not part of
  `/explain-claim`.

## Verification

`pytest tests/test_endpoint.py` posts `artifacts/sample_claim.json` to `/explain-claim` and
checks the response shape and decision polarity. Run this after any change to
`predictor.py`, `embedder.py`, or the graph nodes.
