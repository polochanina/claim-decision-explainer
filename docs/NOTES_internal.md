# Internal notes — what this project actually does

(Internal scratch doc, not part of the submission. For the real deliverables see
`../README.md` for running it and `DESIGN.md` for the full architecture/justification.)

## In one sentence

A FastAPI service takes a raw insurance claim, predicts approve/decline with a
pre-trained CatBoost model, and uses Claude to explain that decision in plain language
to two different audiences — the customer and the claims adjuster.

## Step by step (what happens on one request)

`POST /explain-claim` runs a 4-step LangGraph pipeline:

1. **Predict** (`app/graph/nodes/predict.py`, `app/model/predictor.py`) — the claim's
   free-text `issueDesc` field is embedded live via Voyage (`voyage-4`, 256-dim), then
   combined with the claim's structured fields (device type, retailer, fee, etc.) in the
   exact column order the model was trained on. The pre-trained CatBoost model
   (`artifacts/catboost_model.cbm`) scores the row → a decline probability.
2. **Attribute** (`app/graph/nodes/attribute.py`) — SHAP values are computed for that
   same row so we know *which* factors pushed the decision toward approve or decline
   (e.g. `touchScreen`, `policyStatus`), plus how much the free-text description
   contributed overall.
3. **Explain, fan-out** (`app/graph/nodes/explain.py`, `app/prompts/`) — two parallel
   Claude calls turn the decision + SHAP evidence into:
   - a **customer**-facing explanation: plain language, short, "what happened / why /
     what to do next".
   - an **adjuster**-facing explanation: technical, references the SHAP factors
     directly, and flags it if the free-text description seems to disagree with the
     structured-data signal.
   Both prompts are constrained to *explain* the model's decision, never to second-guess
   or override it.
4. **Assemble** (`app/graph/nodes/assemble.py`) — bundles the decision, approval
   probability, contributing factors, and both explanations into the JSON response
   (`app/schemas.py`).

## Mapping back to the take-home brief

The brief (`[Take Home] Senior GenAI ML Engineer - GenAI-Powered Claim Approval
Agent.docx`) asks for four things. Where each one landed:

| Brief requirement | Status | Where |
|---|---|---|
| ML claim-approval prediction | ✅ built | CatBoost model, trained in `../notebooks/EDA + modeling.ipynb`, served by `app/model/predictor.py` |
| Multi-persona GenAI explanations (≥2 personas) | ✅ built | customer + adjuster personas, `app/graph/nodes/explain.py` + `app/prompts/` |
| Synthetic claim scenario generation | ⏭️ skipped — explicitly optional in the brief | not implemented |
| Basic REST API | ✅ built | `app/main.py` — `POST /explain-claim`, `GET /health` |
| Cloud deployment / MLOps-LLMOps design | 📝 design only, not deployed | written up in `DESIGN.md`; brief allows "a robust local implementation" in lieu of real cloud infra |
| Evaluation & monitoring plan | 📝 design only | written up in `DESIGN.md` §6 |

So: the two **required** GenAI/ML pieces are actually implemented and runnable; the
deployment/MLOps/evaluation sections are a written design (as the brief explicitly
permits when cloud access isn't part of the exercise), and synthetic data generation was
left out since the brief marks it optional.

## Where to look for more detail

- `../README.md` — how to install, run, and test it.
- `DESIGN.md` — architecture diagram, modeling results (AUC/F1), prompt engineering
  strategy, AWS deployment plan, MLOps/LLMOps and evaluation/monitoring writeup.
- `../CLAUDE.md` — the train/serve feature contract (column order, categorical handling,
  embedding spec) — read this before touching `app/model/predictor.py`.
