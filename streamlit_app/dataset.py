from pathlib import Path

import pandas as pd

from streamlit_app.config import CLAIM_REQUEST_COLUMNS


class ClaimSampler:
    def __init__(self, dataset_paths: dict[str, Path]):
        self._dfs = {source: pd.read_excel(path) for source, path in dataset_paths.items()}

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
