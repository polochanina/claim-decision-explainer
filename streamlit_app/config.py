from pathlib import Path

from app.config import PERSONAS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "claim_use_case_dataset.xlsx"
SYNTHETIC_DATASET_PATH = PROJECT_ROOT / "data" / "synthetic_claims_dataset.xlsx"

DATASET_SOURCES = {
    "real": DATASET_PATH,
    "generated": SYNTHETIC_DATASET_PATH,
}

API_BASE_URL = "http://127.0.0.1:8000"
EXPLAIN_ENDPOINT = f"{API_BASE_URL}/explain-claim"

APP_NAME = "Claim Approval Explainer"
APP_ICON = "🛡️"

CLAIM_REQUEST_COLUMNS = [
    "excessFee", "rrp", "productName", "productDesc", "policyStatus",
    "retailerName", "deviceType", "model", "channel", "claimType", "country",
    "turnOnOff", "touchScreen", "frontCamera", "backCamera", "audio", "mic",
    "buttons", "connection", "charging", "issueDesc",
]

PERSONA_LABELS = {
    "customer": "Customer",
    "adjuster": "Claims adjuster",
}
PERSONA_KEYS = list(PERSONAS.keys())

UI_TEXT = {
    "title": "Claim Approval Explainer",
    "caption": "Sample a raw claim and see why the model approved or declined it.",
    "sidebar_header": "Settings",
    "country_label": "Country",
    "all_countries": "All",
    "source_label": "Claim source",
    "source_real": "Real",
    "source_generated": "Generated",
    "persona_label": "Explain as",
    "sample_button": "🎲 Draw a random claim",
    "claim_header": "Raw claim",
    "explain_button": "Explain this claim",
    "spinner_explaining": "Asking the model...",
    "decision_label": "Decision",
    "probability_label": "Approval probability",
    "factors_header": "Top contributing factors",
    "explanation_header": "Explanation",
    "no_claim_warning": "Draw a random claim first.",
    "no_claims_for_country_warning": "No claims found for this country.",
    "no_explanation_for_persona_warning": "No explanation returned for this persona.",
    "error_title": "Something went wrong",
    "decision_approved": "✅ Approved",
    "decision_declined": "❌ Declined",
}
