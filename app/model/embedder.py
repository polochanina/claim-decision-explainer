import numpy as np
import voyageai
from langfuse import Langfuse

from app.config import VOYAGE_EMBED_TRACE_NAME


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str, dim: int, langfuse: Langfuse | None = None):
        self.client = voyageai.Client(api_key)
        self.model = model
        self.dim = dim
        self._langfuse = langfuse

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._langfuse is None:
            resp = self._call_voyage(texts)
            return np.array(resp.embeddings, dtype=np.float32)
        with self._langfuse.start_as_current_observation(
            name=VOYAGE_EMBED_TRACE_NAME, as_type="generation", model=self.model, input=texts
        ) as generation:
            resp = self._call_voyage(texts)
            generation.update(
                output={"dim": self.dim, "count": len(texts)},
                usage_details={"input": getattr(resp, "total_tokens", None)},
            )
            return np.array(resp.embeddings, dtype=np.float32)

    def _call_voyage(self, texts: list[str]) -> voyageai.object.EmbeddingsObject:
        return self.client.embed(
            texts,
            model=self.model,
            input_type="document",
            output_dimension=self.dim,
        )
