import os
from typing import Optional
import requests
from transformers import AutoTokenizer

from .base_summarizer import BaseSummarizationModel

DEFAULT_SUMMARIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a financial analyst summarizing SEC 10-K filing excerpts for "
    "corporate credit risk assessment. Produce concise, factual summaries that "
    "preserve key business details, risks, management information, and financial "
    "context. Do not invent facts."
)


class QwenSummarizationModel(BaseSummarizationModel):
    """RAPTOR-compatible summarizer backed by Qwen2.5-7B-Instruct API."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[any] = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = os.getenv("QWEN_MODEL_NAME", model_name)
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Determine the API URL (defaults to HuggingFace Serverless Inference OpenAI-compatible endpoint)
        raw_url = api_url or os.getenv("QWEN_API_URL") or os.getenv("HF_API_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api-inference.huggingface.co/v1"
        self.api_url = raw_url.rstrip("/")

        
        # Load local tokenizer for token calculations (lightweight, runs on CPU)
        # Use try-except to handle cases where downloading the exact model tokenizer fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            # Fallback to a standard generic tokenizer if name is not found/offline
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

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

        # Call OpenAI-compatible chat completion endpoint
        url = f"{self.api_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # If the API call fails, raise a helpful error
            raise RuntimeError(f"Qwen API summarization failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 150, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.api_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Qwen API generation failed: {e}")


