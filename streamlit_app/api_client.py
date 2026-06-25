import requests


class ExplainClient:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    def explain(self, claim: dict) -> dict:
        response = requests.post(self._endpoint, json=claim, timeout=60)
        response.raise_for_status()
        return response.json()
