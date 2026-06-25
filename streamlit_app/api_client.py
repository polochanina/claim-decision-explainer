import requests


class ExplainClient:
    def __init__(self, base_url: str):
        self._base_url = base_url

    def explain(self, claim: dict) -> dict:
        response = requests.post(f"{self._base_url}/explain-claim", json=claim, timeout=60)
        response.raise_for_status()
        return response.json()

    def check_health(self, timeout: float = 5) -> dict:
        response = requests.get(f"{self._base_url}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
