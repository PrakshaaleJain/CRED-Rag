import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings

from src.models.hf_embedding import HFEmbeddingModel
from src.models.llama_summarizer import LlamaSummarizationModel

# Lazily initialized model wrappers to avoid extra setup on imports
_embedding_model: Optional[HFEmbeddingModel] = None
_summarizer_model: Optional[LlamaSummarizationModel] = None


def get_embedding_model() -> HFEmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HFEmbeddingModel()
    return _embedding_model


def get_summarizer_model() -> LlamaSummarizationModel:
    global _summarizer_model
    if _summarizer_model is None:
        _summarizer_model = LlamaSummarizationModel()
    return _summarizer_model


class LocalHFEmbeddings(Embeddings):
    """LangChain wrapper for Fin-E5 embeddings via local inference."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = get_embedding_model()
        embeddings = model.create_embeddings(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = get_embedding_model()
        return model.create_embedding(text)


def generate_local(prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a response using the local LLaMA summarizer."""
    model = get_summarizer_model()
    return model.generate(prompt, max_tokens=max_new_tokens)
