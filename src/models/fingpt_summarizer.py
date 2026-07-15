"""Backward-compatible alias for the Qwen summarizer used in CRED-RAPTOR."""

from .qwen_summarizer import DEFAULT_SUMMARIZER_MODEL, QwenSummarizationModel

__all__ = ["DEFAULT_SUMMARIZER_MODEL", "QwenSummarizationModel"]
