"""Inference engine: load Xiaomi-OCR-0 (+ optional LoRA adapter) and run page OCR.

Design goals:
- lazy heavy imports so the module imports without torch installed;
- optional PEFT adapter (`adapter_dir`) so QLoRA fine-tuned results load here;
- return a validated structured result via `ocr.schema.make_result`.

Usage:
    from ocr import recognize_page
    res = recognize_page("data/raw/page1.jpg", adapter_dir="outputs/qlora-v1")
"""

from __future__ import annotations

import os
from typing import Optional

from .schema import make_result
from .postprocess import postprocess
from .prompts import SYSTEM_PROMPT, build_messages

DEFAULT_BASE = os.environ.get("OCR_BASE_MODEL", "SeerRay-Lab/Xiaomi-OCR-0")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("OCR_MAX_NEW_TOKENS", "4096"))


class OCREngine:
    """Thin wrapper around a (optionally 4-bit + LoRA) multimodal model."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE,
        adapter_dir: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        trust_remote_code: bool = True,
    ):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None

    # -- lazy load -----------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=self.trust_remote_code
        )

        kwargs = dict(trust_remote_code=self.trust_remote_code)
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = self.device
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = self.device

        model = AutoModelForImageTextToText.from_pretrained(self.base_model, **kwargs)

        if self.adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_dir)

        model.eval()
        self._model = model

    # -- inference -----------------------------------------------------------
    def recognize_page(self, image_path: str, use_few_shot: bool = True) -> dict:
        self._load()
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_messages(use_few_shot)

        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt], images=[image], return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(gen, skip_special_tokens=True)[0]

        pp = postprocess(text)
        model_tag = f"{os.path.basename(self.base_model)}" + (
            f"+{os.path.basename(self.adapter_dir)}" if self.adapter_dir else ""
        )
        return make_result(
            markdown=pp["markdown"],
            structure_ok=pp["structure_ok"],
            latex_issues=pp["latex_issues"],
            tables=pp["stats"]["tables"],
            formulas=pp["stats"]["formulas"],
            model=model_tag,
        )

    def recognize_batch(self, image_paths, use_few_shot: bool = True):
        return [self.recognize_page(p, use_few_shot) for p in image_paths]


# Convenience singleton for the default config.
_default_engine: Optional[OCREngine] = None


def recognize_page(image_path: str, adapter_dir: Optional[str] = None, **kwargs) -> dict:
    """One-shot helper using a module-level default engine."""
    global _default_engine
    if _default_engine is None or adapter_dir is not None:
        eng = OCREngine(adapter_dir=adapter_dir, **kwargs)
        if adapter_dir is None:
            _default_engine = eng
        return eng.recognize_page(image_path)
    return _default_engine.recognize_page(image_path)
