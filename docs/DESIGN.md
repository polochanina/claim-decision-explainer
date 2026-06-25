# Design Document — Claim Approval Agent

## 1. Architecture

```
POST /explain-claim
        │
        ▼
   ┌─────────┐     ┌───────────┐     ┌──────────────────┐     ┌──────────┐
   │ predict │ ──▶ │ attribute │ ──▶ │ explain (fan-out) │ ──▶ │ assemble │ ──▶ response
   └─────────┘     └───────────┘     └──────────────────┘     └──────────┘
                                       ├─ explain_customer
                                       └─ explain_adjuster
```

It's a LangGraph `StateGraph` (`app/graph/build.py`) over one shared `ClaimState` TypedDict
(`app/graph/state.py`). Each node is a small class, built once at startup with its
dependencies — the CatBoost predictor, the Voyage embedder, the Anthropic client — injected
rather than fetched inside the node. See `app/graph/nodes.py`.

**predict** embeds `issueDesc` and `other` live via Voyage (`voyage-4`, 256-dim each,
Matryoshka-truncated — no need for the full width on a dataset this small), builds the
feature row in the exact column order from `artifacts/feature_spec.json`, and runs the
pre-trained CatBoost model (`artifacts/catboost_model.cbm`). Writes `decline_probability`
and `decision` to state.

**attribute** runs CatBoost's own `get_feature_importance(type="ShapValues")` on that same
row and splits the result into per-feature SHAP values (e.g. `touchScreen: -0.071`) plus one
summed contribution per embedded text field — `text_contributions["issueDesc"]` and
`text_contributions["other"]`, each across its own 256 embedding dimensions. This is the
evidence the LLM gets — explanations are grounded in it, not invented.

**explain (fan-out)** runs `explain_customer` and `explain_adjuster` in parallel, both
reading the same upstream state (`decision`, `tabular_shap`, `text_contributions`, the raw
`issueDesc` and `other`) and writing into a shared `persona_explanations` dict through a merge reducer
(`Annotated[dict, merge_dicts]` on `ClaimState`) so neither branch clobbers the other. This
is the one place the graph actually behaves like a graph instead of a linear script.

**assemble** collates `decision`, `approval_probability` (`1 - decline_probability`), the
sorted contributing factors, `text_contributions`, and both persona explanations into the
response, validated by the `ExplanationResponse` Pydantic schema.

### Train/serve contract

The most common way serving silently breaks is a column-order or preprocessing mismatch
with training. `artifacts/feature_spec.json` is the explicit contract for that: exact
`feature_order`, which columns are categorical (`fillna("missing")`, cast to `str`), which
are numeric (NaN left alone — CatBoost handles it natively), and the embedding spec (model,
dimension, input type). It's exported once from the notebook and enforced, not re-derived,
by `app/model/predictor.py`. `ClaimPredictor.__init__` raises at startup if the spec's
feature count doesn't match what the loaded model expects, so a mismatch fails loudly at
boot instead of quietly mispredicting at request time.

One polarity note worth flagging: `target == 1` means **declined**, so `predict_proba[:, 1]`
is P(decline). The API converts that to `approval_probability = 1 - P(decline)` exactly
once, in `predictor.py` — better than risking that inversion getting redone, and gotten
wrong, somewhere downstream in a prompt or the UI.

## 2. ML modeling

From `notebooks/EDA + modeling.ipynb`: a CatBoost classifier on Bolttech's mock device-claim
dataset — 831 rows, 15.7% decline rate, imbalanced enough that I used
`auto_class_weights="Balanced"`.

What got dropped: constant/near-empty columns (`deviceCost`, `relationship`, etc.), date
columns (near-zero correlation with the target once I'd derived duration/age features from
them), and the multi-collinear RRP variants (`balanceRRP`, `oldBalanceRRP`). Categorical
columns use CatBoost's native handling, missing values filled with the literal string
`"missing"`.

Hyperparameters came from Optuna — 30 trials, TPE sampler, over `depth`, `learning_rate`,
`l2_leaf_reg` — with 5-fold stratified CV driving both the search and the early-stopping
iteration count used for the final fit.

Tabular-only, the model is weak: CV AUC 0.673, held-out ROC-AUC 0.618, F1 0.273. That tracks
with the EDA — none of the structured features correlate much with the target (Cramér's V
near zero across the board). Adding the `issueDesc` text embeddings (Voyage `voyage-4`,
256-dim) pushes held-out ROC-AUC to 0.754 and F1 to 0.457 — a real lift, and the empirical
reason Claude reads `issueDesc` directly in the explanation step rather than leaning on the
model's embedding contribution alone: the free text is where most of the decision signal
actually lives.

I reused the same tuned hyperparameters and CV iteration count for the combined model rather
than re-tuning, so the AUC delta isolates what the text itself contributes instead of
conflating it with "found a better model."

The shipped model (`artifacts/catboost_model.cbm`, 532 features) goes one step further and
embeds `other` — the claim's secondary free-text notes field, often empty — the same way as
`issueDesc`, on the theory that any free-text signal not already captured by `issueDesc`
is worth letting the model see rather than discarding. I haven't re-measured held-out
ROC-AUC/F1 for this issueDesc+other combination specifically (the notebook's ablation cell
only compares tabular-only against tabular+issueDesc); treat the 0.754/0.457 numbers above
as the issueDesc-only baseline the `other` embedding builds on top of, not as the current
model's number.

## 3. GenAI: prompt engineering strategy

Two persona prompts (`app/prompts/customer.txt`, `app/prompts/adjuster.txt`), same template
inputs — decision, top SHAP factors, raw `issueDesc` — different register and depth:

- **Customer**: plain language, no model jargon. What happened, the one or two factors that
  actually mattered, what to do next. Capped around 150 words.
- **Adjuster**: technical, references the SHAP factors and their direction by name, and is
  explicitly told to compare the structured-data signal against each text signal
  (`issueDesc` and `other` are scored and surfaced separately) — any disagreement, including
  the two text sources disagreeing with each other, gets surfaced as a named **review flag**,
  not quietly resolved one way or the other.

The constraint that matters most in both prompts: explain the decision, never second-guess
it. If the free text reads as more or less deniable than the structured evidence alone
suggests, the adjuster prompt flags that tension for a human to look at — it doesn't get to
propose a different outcome. The point is to keep the LLM acting as a narrator of the
model's SHAP evidence, not a second decision-maker. The realistic hallucination risk in a
system like this isn't made-up facts, it's the LLM quietly overriding the model's call, and
this constraint is the direct guard against that.

Claude reads both `issueDesc` and `other` as-is — Swedish in the sample data, mixed
languages elsewhere — rather than through a translation step first, since the AUC numbers
above already show free text is where the signal is.

### Targeted synthetic claim scenarios (implemented)

Built in `notebooks/synthetic_scenario_generation.py` (+ paired `.ipynb`), run offline
against the real Claude + Voyage + `ClaimPredictor` stack. Output lands in `data/`, never
folded into training data and never read by the app. Two modes, because
"underrepresented" and "actually challenges the model" need different prompts — forcing a
clean label and asking for an ambiguous one are opposite instructions.

**Segment coverage** (`data/synthetic_claims_dataset.xlsx`). The raw dataset
(`data/claim_use_case_dataset.xlsx`, before cleaning) has a handful of low-support
segments with decline rates well above average that the 831-row training set barely
covers: `deviceType=WEARABLES` (n=30, 50.0% decline), `LAPTOP` (n=35, 28.6%),
`claimType=Theft` (n=98, 20.4%), `country=FI` (n=136, 16.2%) — those are the targets, not
invented categories. One Claude call per segment: output pinned to the raw dataset's
columns, categorical values restricted to ones actually seen in that column, grounded by
2-3 real anonymized rows as few-shot reference, varying the narrative *reasoning* behind
the denial while asserting `status = "Declined"` for every row. A sanity-check pass runs
every row through the real `ClaimPredictor` and reports how many agree with that asserted
label — disagreement doesn't mean a row is broken, it means the model itself wouldn't have
declined that particular scenario, worth knowing if this set is ever read as eval ground
truth rather than denial-pattern coverage.

**Boundary-challenging** (`data/synthetic_borderline_claims_dataset.xlsx`). The set above
always asserts `Declined`, so none of those rows actually challenge the model — they're
easy cases written to a known answer. This mode does the opposite: `status` is left
undetermined, and the prompt asks for tension between the structured fields and the
`issueDesc` narrative (a low-risk structured profile paired with a vague/inconsistent
narrative, or the reverse), explicitly ruling out the obvious decline tells the first
prompt asks for. The loop closes against the real model instead of trusting the LLM's own
sense of what's ambiguous: each segment oversamples 6 candidates, scores every one with
`ClaimPredictor.predict`, and keeps only the ones landing in `(0.4, 0.6)` decline
probability — typically 2-3 of 6 survive. Kept rows carry the model's own predicted
probability/decision instead of an asserted label, since the goal is finding claims the
model is unsure about, not manufacturing ground truth it would disagree with.

## 4. Deployment (AWS)

This is a low-QPS workload — two LLM calls per request, one small model file — so I kept the
target deliberately minimal:

- **Compute**: ECS Fargate, running the FastAPI app in a container. (Lambda behind API
  Gateway is the cheaper option if traffic stays low/bursty and the CatBoost+Voyage
  cold-start hit is acceptable — Fargate just stays warm, which matters more once two
  sequential LLM calls are already in the latency budget.)
- **Storage**: `artifacts/` (model + feature spec) in S3, pulled at build or startup — keeps
  the model versioned independently of the app image.
- **Secrets**: AWS Secrets Manager for `ANTHROPIC_API_KEY`/`VOYAGE_API_KEY`, injected as env
  vars at task startup, never baked into the image.
- **Networking**: API Gateway or an ALB in front of Fargate for the public endpoint, the
  service itself in a private subnet.

Why this and not more: traffic is low and the two sequential LLM calls already dominate
response time, so there's no case yet for multi-region or heavy autoscaling. The thing
actually worth getting right is keeping the model artifact and the secrets out of the image
and versionable independently of the code — that's where this design puts the effort.

## 5. MLOps / LLMOps

- **Model versioning**: `catboost_model.cbm` and `feature_spec.json` always move together —
  any retrain re-exports both, never just one. In production that pair would live in S3 with
  a version tag the deployed task definition points at. Locally, the notebook also logs each
  training run (params, metrics, the model artifact) to MLflow (`sqlite:///mlflow.db` at the
  repo root, committed alongside `mlruns/` so a reviewer can run
  `mlflow ui --backend-store-uri sqlite:///mlflow.db` without retraining) and registers the
  served model as `claim-approval-catboost` in the MLflow Model Registry, with the run id and
  registered version embedded into `feature_spec.json` so `ClaimPredictor` can report its own
  lineage (surfaced at `/health`) without the app ever connecting to MLflow at serve time.
- **Prompt versioning**: `app/prompts/*.txt` are plain files in version control, not inline
  strings, so a prompt change is just an ordinary, reviewable diff. Each persona prompt is
  also registered as a versioned entry (`claim-explainer-customer`, `claim-explainer-adjuster`)
  in the MLflow Prompt Registry, tagged with the run id of the model version it was logged
  alongside — an explicit, queryable pairing between a prompt revision and the model version
  it shipped with, on top of the git history already tracking prompt text changes.
- **CI/CD** (`.github/workflows/ci.yml`): `pytest` already runs on every push/PR, exercising
  the live model and prompts end-to-end via `tests/test_endpoint.py` — so a model or prompt
  change that breaks the contract fails the build before merge. Not built: image build/push
  and an IaC-managed deploy step (Terraform/CDK provisioning the Fargate task, S3 bucket, and
  Secrets Manager entries from section 4) on merge to main.
- **Monitoring, what's actually wired up**: every `/explain-claim` request opens a Langfuse
  trace (`app/observability.py`, optional and `None`-safe if no keys are set) with a root
  span plus three child generation spans — `voyage-embed`, `explain-customer`,
  `explain-adjuster`. Each one reports model and token usage, so the Langfuse dashboard gets
  cost and latency for both personas for free. The root span also carries `decision` and
  `approval_probability`, so there's a per-request decision log sitting next to the GenAI
  trace — enough to eyeball recent decline-rate trends, though it's not a real drift
  pipeline.
- **What's not wired up**: model drift (comparing live decline-rate and feature
  distributions against the training distribution and alerting on divergence — the decision
  log above is a starting point for this, not a substitute) and hallucination/contradiction
  auditing (periodically sampling logged explanations against `decision` — straightforward
  to add later since both already live in the same Langfuse trace, just not automated yet).

## 6. Evaluation & monitoring

**ML metrics**: ROC-AUC and F1 on the held-out set (numbers above), tracked over time as the
model gets retrained on more claims. SHAP factor stability across retrains is a useful
secondary signal — if the dominant factors swing a lot between retrains, that's worth
investigating before redeploying.

**GenAI output metrics**: cost, tokens, and latency per persona call already come for free
from the Langfuse traces in section 5 — nothing extra needed beyond what's wired into
`explain.py` and `embedder.py`. Structural validity is enforced synchronously by
`ExplanationResponse` (Pydantic) — a malformed response is a hard failure, not a quality
metric. Future scope: a periodic LLM-as-judge or human-reviewed sample (DeepEval/Ragas
faithfulness checks, or a custom G-Eval-style check) verifying register fits the persona,
the adjuster explanation flags text/tabular tension when SHAP shows one, and no explanation
contradicts `decision` — the one failure mode the prompt constraint exists to prevent. Not
built for this submission; the brief scopes this as a design plan, not a shipped pipeline.

**Responsible AI**: for fairness, track decline rate and explanation outcomes against the
proxy attributes available — `country`, `retailer`, `channel` — to catch disparate impact,
since the EDA already shows decline patterns vary by `country`. For transparency, every
explanation traces back to specific SHAP values in the same response
(`contributing_factors`) instead of being a free-floating LLM narrative — that's what
actually makes "explain, don't override" auditable rather than just a promise in the prompt.
