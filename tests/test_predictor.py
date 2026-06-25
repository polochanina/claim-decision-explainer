import json

import pytest

from app.config import FEATURE_SPEC_PATH, MODEL_PATH
from app.model.embedder import VoyageEmbedder
from app.model.predictor import ClaimPredictor


def test_feature_spec_mismatch_raises(tmp_path):
    spec = json.loads(FEATURE_SPEC_PATH.read_text())
    spec["feature_order"] = spec["feature_order"][:-1]
    bad_spec_path = tmp_path / "feature_spec.json"
    bad_spec_path.write_text(json.dumps(spec))

    embedder = VoyageEmbedder(api_key="test", model="voyage-4", dim=256)

    with pytest.raises(RuntimeError, match="out of sync"):
        ClaimPredictor(embedder, MODEL_PATH, bad_spec_path)
