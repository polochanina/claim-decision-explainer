from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

MODEL_PATH = ARTIFACTS_DIR / "catboost_model.cbm"
FEATURE_SPEC_PATH = ARTIFACTS_DIR / "feature_spec.json"
SAMPLE_CLAIM_PATH = ARTIFACTS_DIR / "sample_claim.json"

VOYAGE_MODEL = "voyage-4"
VOYAGE_OUTPUT_DIM = 256

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 1024

PERSONAS = {
    "customer": PROMPTS_DIR / "customer.txt",
    "adjuster": PROMPTS_DIR / "adjuster.txt",
}

LANGUAGES = {
    "EN": "English",
    "NL": "Dutch",
    "SV": "Swedish",
    "FI": "Finnish",
}
DEFAULT_LANGUAGE = "EN"

DECISION_THRESHOLD = 0.5

EXPLAIN_CLAIM_TRACE_NAME = "explain-claim"
VOYAGE_EMBED_TRACE_NAME = "voyage-embed"
