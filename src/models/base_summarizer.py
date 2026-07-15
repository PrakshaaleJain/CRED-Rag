from abc import ABC, abstractmethod


class BaseSummarizationModel(ABC):
    """RAPTOR-compatible base class for abstractive summarization models."""

    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """Summarize the given context and return the summary text."""
