#!/usr/bin/env python
# coding: utf-8

# # Synthetic claim scenario generation
#
# Two modes for generating fake claims with Claude, offline only -- outputs are saved to
# `data/` and never used by the app or for retraining.
#
# 1. **Segment coverage** -- claims for rare, high-decline segments (`WEARABLES`, `LAPTOP`,
#    `TABLET`, `Theft`, `Liquid Damage`, `country=FI`), always labeled `Declined`, to pad out
#    denial examples in segments the real data barely covers.
# 2. **Borderline** -- claims with no asserted label; only ones the real model scores near
#    50/50 decline probability are kept, so the set actually tests the model instead of
#    telling it the answer.

# In[ ]:


import json
import re
import sys
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

sys.path.insert(0, str(Path.cwd().parent))

from app.secrets_utils import get_secrets
from app.config import CLAUDE_MODEL
from app.model.embedder import VoyageEmbedder
from app.model.predictor import ClaimPredictor


# In[ ]:


DATA_PATH = "../data/claim_use_case_dataset.xlsx"
df = pd.read_excel(DATA_PATH)

TARGET_SEGMENTS = [
    ("deviceType", "WEARABLES"),
    ("deviceType", "LAPTOP"),
    ("deviceType", "TABLET"),
    ("claimType", "Theft"),
    ("claimType", "Liquid Damage"),
    ("country", "FI"),
]
ROWS_PER_SEGMENT = 3


# In[ ]:


secrets = get_secrets()
client = Anthropic(api_key=secrets["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = (
    "You write realistic, structurally valid insurance claim records as JSON. "
    "Output ONLY a JSON array, no prose, no markdown fences."
)


def build_prompt(columns: list[str], examples: list[dict], column: str, value: str, n: int) -> str:
    return f"""Generate {n} synthetic insurance claim records targeting a DECLINED claim where {column} == "{value}".
Each record must be a JSON object with exactly these keys: {columns}.
Use only categorical values consistent with the real examples below. Numeric/checkbox
fields may be null. Vary the issueDesc narrative reasoning for the denial (e.g. policy
exclusion, inconsistent story, pre-existing damage) while keeping it a plausible {value}
claim, written in the same language/style as the examples. status must be "Declined" for
every record.

Real examples for grounding (style/format only, do not copy):
{json.dumps(examples, ensure_ascii=False, indent=2, default=str)}"""


def generate_synthetic_rows(column: str, value: str, n: int = ROWS_PER_SEGMENT) -> list[dict]:
    examples = df[df[column] == value].head(3).fillna("").to_dict(orient="records")
    prompt = build_prompt(list(df.columns), examples, column, value, n)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = re.sub(r"^```(json)?|```$", "", response.content[0].text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


# In[ ]:


rows = []
for column, value in TARGET_SEGMENTS:
    rows.extend(generate_synthetic_rows(column, value))

synthetic_df = pd.DataFrame(rows, columns=df.columns)
synthetic_df.shape


# In[ ]:


OUTPUT_PATH = "../data/synthetic_claims_dataset.xlsx"
synthetic_df.to_excel(OUTPUT_PATH, index=False)
OUTPUT_PATH


# ## Sanity check
#
# Score the segment-coverage rows with the real `ClaimPredictor`. Their `Declined` label
# was asserted by the prompt, not predicted -- so a mismatch below doesn't mean a row is
# bad, just that the model wouldn't have declined that scenario on its own. Don't treat
# this set as ground truth.

# In[ ]:


embedder = VoyageEmbedder(secrets["VOYAGE_API_KEY"], model="voyage-4", dim=256)
predictor = ClaimPredictor(
    embedder=embedder,
    model_path=Path("../artifacts/catboost_model.cbm"),
    feature_spec_path=Path("../artifacts/feature_spec.json"),
)

agrees = 0
for claim in rows:
    row = predictor.build_feature_row(claim)
    probability, decision = predictor.predict(row)
    agrees += decision == "declined"
    print(claim["deviceType"], claim["claimType"], claim["country"], "->", decision, f"{probability:.2%}")

print(f"\n{agrees}/{len(rows)} synthetic rows agree with the asserted Declined label")


# ## Borderline scenario generation
#
# The set above is all easy cases -- every row is written to a known answer. Here the
# label is left blank, and the prompt asks for tension between the structured fields and
# the `issueDesc` narrative instead. Each candidate is scored with the real
# `ClaimPredictor`, and only the ones landing near the decision boundary are kept, with the
# model's own prediction attached.

# In[ ]:


PROBABILITY_BAND = (0.4, 0.6)
CANDIDATES_PER_SEGMENT = 6
ROWS_PER_BORDERLINE_SEGMENT = 3

BORDERLINE_SYSTEM_PROMPT = (
    "You write realistic, structurally valid insurance claim records as JSON. "
    "Output ONLY a JSON array, no prose, no markdown fences."
)


def build_borderline_prompt(columns: list[str], examples: list[dict], column: str, value: str, n: int) -> str:
    return f"""Generate {n} synthetic insurance claim records for a {value} {column} claim where it is
genuinely AMBIGUOUS whether the claim would be approved or declined.
Each record must be a JSON object with exactly these keys: {columns}.
Use only categorical values consistent with the real examples below. Numeric/checkbox
fields may be null. Create deliberate tension between the structured fields and the
issueDesc narrative: pair a low-risk structured profile (active policy, ordinary {value}
claim) with a vague or inconsistent narrative, OR a higher-risk structured profile with an
unusually clear and credible narrative. Do not use obvious decline tells (explicit
exclusion admissions, contradictory dates, stated pre-existing damage) and do not write an
obviously clean approval either -- a claims adjuster reading this should genuinely be
unsure. Leave status null; that determination is not yours to make.

Real examples for grounding (style/format only, do not copy):
{json.dumps(examples, ensure_ascii=False, indent=2, default=str)}"""


def generate_borderline_candidates(column: str, value: str, n: int = CANDIDATES_PER_SEGMENT) -> list[dict]:
    examples = df[df[column] == value].head(3).fillna("").to_dict(orient="records")
    prompt = build_borderline_prompt(list(df.columns), examples, column, value, n)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=BORDERLINE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = re.sub(r"^```(json)?|```$", "", response.content[0].text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


# In[ ]:


borderline_rows = []
for column, value in TARGET_SEGMENTS:
    candidates = generate_borderline_candidates(column, value)
    kept = 0
    for claim in candidates:
        if kept >= ROWS_PER_BORDERLINE_SEGMENT:
            break
        row = predictor.build_feature_row(claim)
        probability, decision = predictor.predict(row)
        if PROBABILITY_BAND[0] <= probability <= PROBABILITY_BAND[1]:
            claim["predicted_decline_probability"] = probability
            claim["predicted_decision"] = decision
            borderline_rows.append(claim)
            kept += 1
    print(column, value, f"-> {kept}/{ROWS_PER_BORDERLINE_SEGMENT} candidates landed in {PROBABILITY_BAND}")


# In[ ]:


borderline_df = pd.DataFrame(
    borderline_rows, columns=list(df.columns) + ["predicted_decline_probability", "predicted_decision"]
)
borderline_df.shape


# In[ ]:


BORDERLINE_OUTPUT_PATH = "../data/synthetic_borderline_claims_dataset.xlsx"
borderline_df.to_excel(BORDERLINE_OUTPUT_PATH, index=False)
BORDERLINE_OUTPUT_PATH

