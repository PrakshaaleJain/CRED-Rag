from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base_embedding import BaseEmbeddingModel


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.clamp(min=1e-9).sum(1)


class HFEmbeddingModel(BaseEmbeddingModel):
    """Local HuggingFace encoder for RAPTOR clustering and retrieval."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def create_embedding(self, text: str) -> List[float]:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        embedding = _mean_pooling(outputs, encoded["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array([self.create_embedding(text) for text in texts])
