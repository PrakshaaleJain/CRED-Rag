import os
from typing import List, Optional
import numpy as np
import requests
from transformers import AutoTokenizer

from .base_embedding import BaseEmbeddingModel


class HFEmbeddingModel(BaseEmbeddingModel):
    """RAPTOR-compatible embedding model wrapper using API inferences."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = os.getenv("BGE_MODEL_NAME", model_name)
        self.max_length = max_length
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or os.getenv("BGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Determine the API URL (defaults to HuggingFace Serverless Inference OpenAI-compatible endpoint)
        raw_url = api_url or os.getenv("BGE_API_URL") or os.getenv("HF_API_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api-inference.huggingface.co/v1"
        self.api_url = raw_url.rstrip("/")


        # Keep tokenizer locally for token counting / split calculations (CPU only, fast)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def create_embedding(self, text: str) -> List[float]:
        # Call OpenAI-compatible /embeddings endpoint
        url = f"{self.api_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "input": text,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            return res_json["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"BGE API embedding creation failed: {e}")

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        # Call OpenAI-compatible /embeddings endpoint for batch
        url = f"{self.api_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            # Parse embeddings in order
            embeddings = [item["embedding"] for item in res_json["data"]]
            return np.array(embeddings)
        except Exception as e:
            # Fallback to individual calls if batch API fails
            print(f"Batch embedding failed: {e}. Falling back to sequential calls.")
            return np.array([self.create_embedding(text) for text in texts])

