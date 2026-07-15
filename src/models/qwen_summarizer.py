from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_summarizer import BaseSummarizationModel

DEFAULT_SUMMARIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a financial analyst summarizing SEC 10-K filing excerpts for "
    "corporate credit risk assessment. Produce concise, factual summaries that "
    "preserve key business details, risks, management information, and financial "
    "context. Do not invent facts."
)


class QwenSummarizationModel(BaseSummarizationModel):
    """RAPTOR-compatible summarizer backed by Qwen2.5-7B-Instruct."""

    def __init__(
        self,
        model_name: str = DEFAULT_SUMMARIZER_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if self.device != "cuda" or model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)

        self.model.eval()

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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
