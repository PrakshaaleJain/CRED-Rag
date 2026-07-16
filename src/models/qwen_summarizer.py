import os
from typing import Optional
import torch
from transformers import AutoTokenizer, pipeline

from .base_summarizer import BaseSummarizationModel

DEFAULT_SUMMARIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a financial analyst summarizing SEC 10-K filing excerpts for "
    "corporate credit risk assessment. Produce concise, factual summaries that "
    "preserve key business details, risks, management information, and financial "
    "context. Do not invent facts."
)


class QwenSummarizationModel(BaseSummarizationModel):
    """RAPTOR-compatible summarizer backed by a local Qwen model on GPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        **kwargs
    ) -> None:
        self.model_name = os.getenv("QWEN_MODEL_NAME", model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model_kwargs = {"torch_dtype": torch.float16}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        device_id = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading {self.model_name} locally on {'GPU' if device_id == 0 else 'CPU'}...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            model_kwargs=model_kwargs,
            device=device_id
        )

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

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()

    def generate(self, prompt: str, max_tokens: int = 150, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()

