import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from app.config import DECISION_THRESHOLD
from app.model.embedder import VoyageEmbedder


class ClaimPredictor:
    def __init__(self, embedder: VoyageEmbedder, model_path: Path, feature_spec_path: Path):
        self._embedder = embedder

        self._model = CatBoostClassifier()
        self._model.load_model(str(model_path))

        spec = json.loads(feature_spec_path.read_text())
        self._feature_order = spec["feature_order"]
        self._tabular_columns = spec["tabular_columns"]
        self._cat_features = spec["cat_features"]
        self._cat_feature_set = set(self._cat_features)
        self._text_column = spec["preprocessing"]["text_column"]
        self._mlflow_lineage = spec.get("mlflow", {})

        if len(self._feature_order) != len(self._model.feature_names_):
            raise RuntimeError(
                "feature_spec.json is out of sync with catboost_model.cbm: "
                f"spec has {len(self._feature_order)} features, model expects "
                f"{len(self._model.feature_names_)}. Re-export both from the notebook."
            )

    def build_feature_row(self, claim: dict) -> pd.DataFrame:
        row: dict[str, object] = {}
        for col in self._tabular_columns:
            value = claim.get(col)
            if col in self._cat_feature_set:
                row[col] = "missing" if value is None else str(value)
            else:
                row[col] = value

        text = (claim.get(self._text_column) or "").strip() or " "
        embedding = self._embedder.embed([text])[0]
        for i, value in enumerate(embedding):
            row[f"emb_{self._text_column}_{i}"] = float(value)

        return pd.DataFrame([row], columns=self._feature_order)

    def predict(self, row: pd.DataFrame) -> tuple[float, str]:
        decline_probability = float(self._model.predict_proba(row)[0, 1])
        decision = "declined" if decline_probability >= DECISION_THRESHOLD else "approved"
        return decline_probability, decision

    def shap_values(self, row: pd.DataFrame) -> tuple[dict[str, float], float]:
        pool = Pool(row, cat_features=self._cat_features)
        shap_row = self._model.get_feature_importance(pool, type="ShapValues")[0, :-1]

        tabular_shap = {}
        text_contribution = 0.0
        for column, contribution in zip(row.columns, shap_row):
            if column.startswith(f"emb_{self._text_column}_"):
                text_contribution += float(contribution)
            else:
                tabular_shap[column] = float(contribution)

        return tabular_shap, text_contribution

    @property
    def model_lineage(self) -> dict:
        return self._mlflow_lineage
