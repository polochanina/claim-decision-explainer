from pathlib import Path

import ftfy
import pandas as pd

from streamlit_app.config import CLAIM_REQUEST_COLUMNS


class ClaimSampler:
    def __init__(self, dataset_paths: dict[str, Path]):
        self._dfs = {source: self._load_repaired(path) for source, path in dataset_paths.items()}

    @staticmethod
    def _load_repaired(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path)
        for column in df.select_dtypes(include="object").columns:
            df[column] = df[column].map(lambda value: ftfy.fix_text(value) if isinstance(value, str) else value)
        return df

    def countries(self, source: str) -> list[str]:
        return sorted(self._dfs[source]["country"].dropna().unique().tolist())

    def sample(self, source: str, country: str | None) -> dict:
        df = self._dfs[source]
        pool = df if country is None else df[df["country"] == country]
        if pool.empty:
            return {}
        row = pool.sample(n=1).iloc[0]
        claim = row[CLAIM_REQUEST_COLUMNS].where(row[CLAIM_REQUEST_COLUMNS].notna(), None)
        return claim.to_dict()
