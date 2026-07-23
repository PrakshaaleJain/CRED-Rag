import os
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .base_embedding import BaseEmbeddingModel


class HFEmbeddingModel(BaseEmbeddingModel):
    """RAPTOR-compatible embedding model wrapper using local SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "yixuantt/Fin-E5",
        max_length: int = 512,
        **kwargs
    ) -> None:
        self.model_name = os.getenv("FINE5_MODEL_NAME", model_name)
        self.max_length = max_length

        # SentenceTransformer automatically uses CUDA if available
        print(f"Loading local SentenceTransformer model ({self.model_name})...")
        self.model = SentenceTransformer(self.model_name)

    def create_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
