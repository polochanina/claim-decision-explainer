import json

from fastapi.testclient import TestClient

from app.config import SAMPLE_CLAIM_PATH
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_explain_claim_happy_path():
    claim = json.loads(SAMPLE_CLAIM_PATH.read_text())

    response = client.post("/explain-claim", json=claim)

    assert response.status_code == 200
    body = response.json()

    assert body["decision"] in ("approved", "declined")
    assert 0.0 <= body["approval_probability"] <= 1.0
    assert {"customer", "adjuster"} == {e["persona"] for e in body["explanations"]}
    for explanation in body["explanations"]:
        assert len(explanation["explanation"]) > 0
