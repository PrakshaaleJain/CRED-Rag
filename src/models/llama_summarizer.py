import os
import requests
from typing import Optional

from .base_summarizer import BaseSummarizationModel

DEFAULT_SUMMARIZER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a financial analyst summarizing SEC 10-K filing excerpts for "
    "corporate credit risk assessment. Produce concise, factual summaries that "
    "preserve key business details, risks, management information, and financial "
    "context. Retain all exact percentages, currency figures, dates, and "
    "covenant thresholds verbatim from the source text. Do not invent facts."
)


class LlamaSummarizationModel(BaseSummarizationModel):
    """RAPTOR-compatible summarizer backed by a local vLLM server."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        **kwargs
    ) -> None:
        self.model_name = os.getenv("LLAMA_MODEL_NAME", model_name)
        self.api_url = os.getenv("VLLM_API_URL", VLLM_API_URL)
        print(f"Connecting to vLLM server at {self.api_url} for model {self.model_name}...")

    def summarize(self, context: str, max_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Write a summary of the following text, including as many "
                    f"key details as possible:\n\n{context}"
                ),
            },
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Error] vLLM API connection failed: {e}")
            print(f"Ensure your vLLM server is running at {self.api_url}")
            return ""

    def generate(self, prompt: str, max_tokens: int = 150, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Error] vLLM API connection failed: {e}")
            return ""
