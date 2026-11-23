"""Inference engine: load Xiaomi-OCR-0 (+ optional LoRA adapter) and run page OCR.

v2 protocol (see inference_protocol_v2.json):
- prompt is *training-aligned*: no system message, one user turn with the image
  content block followed by "\\n" + instruction, so the token sequence matches
  what ms-swift's qwen3_5 template produced during training (verified by
  scripts/probe_protocol.py: prompt_identical=True).
- image pixel budget is explicit (`max_pixels`), applied through
  processor.image_processor.size["longest_edge"] so both the base model and the
  fine-tuned model see the same resolution (training used max_pixels=1024000).
- every run records enough evidence to reproduce and audit it: image sha256,
  prompt hash, grid/patch/visual-token counts, generated tokens, stop reason,
  repetition flags, timing and memory.

Usage:
    from ocr import recognize_page
    res = recognize_page("data/raw/page1.jpg", adapter_dir="outputs/qlora-v1/.../checkpoint-4")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from .paths import resolve_image_path
from .postprocess import normalize_output, postprocess, strip_outer_fence
from .prompts import INSTRUCTION, build_messages
from .schema import make_result

DEFAULT_BASE = os.environ.get("OCR_BASE_MODEL", "SeerRay-Lab/Xiaomi-OCR-0")
# 1024 is chosen from non-test evidence: the longest *training/validation*
# reference is 611 supervised tokens (see outputs/protocol/supervised_tokens.json),
# so 512 cannot cover a normal full page while 1024 leaves ~1.6x headroom.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("OCR_MAX_NEW_TOKENS", "1024"))
DEFAULT_MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "1024000"))
PROTOCOL_VERSION = "v2"

# Published repetition rules (see inference_protocol_v2.json -> repetition_rules)
REP_LINE_MIN = 5          # same non-empty line repeated >=5 times in a row
REP_UNIT = r"(.{2,40}?)\1{4,}"   # a 2-40 char unit repeated >=5 times consecutively
REP_NGRAM = 10
REP_NGRAM_MIN_COUNT = 20
REP_NGRAM_MIN_COVERAGE = 0.25
# Degeneration often only increments an index, so raw line equality never fires
# (observed on checkpoint-4: W_{12}..W_{33} = identical formula). Masking digits
# and requiring 5 consecutive identical *templates* catches that shape. Table rows
# are excluded because legitimately-uniform table rows would false-positive.
REP_TEMPLATE_MIN = 5
_DIGITS_RE = re.compile(r"\d+")

_NVIDIA_SMI = ["nvidia-smi", "--query-gpu=memory.used,memory.total",
               "--format=csv,noheader,nounits"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gpu_memory_used_mib() -> Optional[List[int]]:
    """WDDM 'memory.used' for the GPU: includes *all* processes, not just ours."""
    try:
        out = subprocess.run(_NVIDIA_SMI, capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[0]
        used, total = [int(x.strip()) for x in line.split(",")]
        return [used, total]
    except Exception:
        return None


def detect_repetition(text: str) -> Dict:
    """Deterministic repetition flags. Returns counts + one example per rule."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    max_line_run, line_sample = 0, None
    run, prev = 0, None
    for l in lines:
        run = run + 1 if l == prev else 1
        if run > max_line_run:
            max_line_run, line_sample = run, l
        prev = l

    unit_max, unit_sample = 0, None
    # Table rows are excluded: `| --- | --- | ...` in a legitimately wide table is a
    # repeated unit and would otherwise false-positive (observed on IMG_..._235946).
    prose = "\n".join(l for l in text.splitlines() if not l.strip().startswith("|"))
    for m in re.finditer(REP_UNIT, prose, flags=re.S):
        n = len(m.group(0)) // max(1, len(m.group(1)))
        if n > unit_max:
            unit_max, unit_sample = n, m.group(1)

    # digit-masked line templates (index-increment degeneration)
    tmpl_max, tmpl_sample, tmpl_raw = 0, None, None
    trun, tprev = 0, None
    for l in lines:
        if l.startswith("|"):          # tables are intentionally excluded
            trun, tprev = 0, None
            continue
        t = _DIGITS_RE.sub("#", l)
        trun = trun + 1 if t == tprev else 1
        if trun > tmpl_max:
            tmpl_max, tmpl_sample, tmpl_raw = trun, t, l
        tprev = t

    grams = [text[i:i + REP_NGRAM] for i in range(0, max(0, len(text) - REP_NGRAM + 1))]
    top_gram, top_count, coverage = None, 0, 0.0
    if grams:
        from collections import Counter
        cnt = Counter(grams)
        top_gram, top_count = cnt.most_common(1)[0]
        coverage = min(1.0, top_count * REP_NGRAM / max(1, len(text)))

    flags = {
        "line_run": {"max_consecutive": max_line_run, "threshold": REP_LINE_MIN,
                     "flagged": max_line_run >= REP_LINE_MIN, "sample": (line_sample or "")[:60]},
        "unit_run": {"max_consecutive": unit_max, "threshold": 5,
                     "flagged": unit_max >= 5, "sample": (unit_sample or "")[:40],
                     "rule": "去掉表格行后，某 2-40 字符单元连续重复 >=5 次"},
        "template_line_run": {"max_consecutive": tmpl_max, "threshold": REP_TEMPLATE_MIN,
                              "flagged": tmpl_max >= REP_TEMPLATE_MIN,
                              "sample": (tmpl_sample or "")[:60],
                              "raw_sample": (tmpl_raw or "")[:60],
                              "rule": "连续 >=5 行仅数字不同（数字替换为 # 后相同），表格行不参与"},
        "ngram": {"n": REP_NGRAM, "top_count": top_count, "coverage": round(coverage, 4),
                  "thresholds": [REP_NGRAM_MIN_COUNT, REP_NGRAM_MIN_COVERAGE],
                  "flagged": top_count >= REP_NGRAM_MIN_COUNT and coverage >= REP_NGRAM_MIN_COVERAGE,
                  "sample": (top_gram or "")[:40]},
    }
    flags["detected"] = any(v["flagged"] for k, v in flags.items() if isinstance(v, dict))
    return flags


class OCREngine:
    """Thin wrapper around a (optionally 4-bit + LoRA) multimodal model."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE,
        adapter_dir: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        use_system_prompt: bool = False,
        trust_remote_code: bool = True,
    ):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.use_system_prompt = use_system_prompt
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None
        self._adapter_meta: Dict = {}
        self.last_run: Dict = {}

    # -- lazy load -----------------------------------------------------------
    def _apply_pixel_budget(self):
        """Set the pixel budget through the image processor (training parity)."""
        ip = self._processor.image_processor
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            size = dict(size)
            size["longest_edge"] = self.max_pixels
            if "shortest_edge" not in size:
                if "height" in size and "width" in size:
                    size["shortest_edge"] = min(size["height"], size["width"])
                else:
                    # Qwen-family image processors use a shortest-edge floor.
                    # Keep an explicit fallback instead of KeyError-ing on a
                    # processor whose size dict only carries longest_edge.
                    size["shortest_edge"] = 56
            ip.size = size
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            ip.size = {"shortest_edge": min(size), "longest_edge": self.max_pixels}
        else:
            ip.size = {"shortest_edge": 56, "longest_edge": self.max_pixels}

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=self.trust_remote_code
        )
        self._apply_pixel_budget()

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

        self._model = AutoModelForImageTextToText.from_pretrained(self.base_model, **kwargs)
        if self.adapter_dir:
            self.attach_adapter(self.adapter_dir)
        self._model.eval()

    # PEFT renames keys on the way out: it inserts the adapter name
    # (".default.") and may/may not keep the "base_model.model." prefix.
    # Both sides are normalised to a bare module path + suffix before matching.
    @staticmethod
    def _norm_lora_key(key: str) -> str:
        k = key
        for pref in ("base_model.model.", "model."):
            if k.startswith(pref):
                k = k[len(pref):]
                break
        k = k.replace(".default.", ".")
        return k

    def attach_adapter(self, adapter_dir: str) -> Dict:
        """Attach a PEFT adapter and *prove* the weights landed correctly.

        Verification is four independent checks, none of which is "it loaded":
        1. every on-disk tensor maps to a live parameter after key normalisation;
        2. the live values equal the on-disk values (max abs diff);
        3. PEFT's own loader reports zero missing / zero unexpected keys;
        4. the LoRA modules actually live inside the language model, and the
           trained lora_B tensors are non-zero (a zero lora_B would mean the
           adapter is functionally a no-op even though it loaded).
        """
        import torch
        from peft import PeftModel, set_peft_model_state_dict
        from safetensors.torch import load_file

        self.adapter_dir = adapter_dir
        self._model = PeftModel.from_pretrained(self._model, adapter_dir, is_trainable=False)
        self._model.eval()

        cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"))
        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.isfile(weights_path):
            disk = load_file(weights_path)
        else:
            bin_path = os.path.join(adapter_dir, "adapter_model.bin")
            if not os.path.isfile(bin_path):
                raise FileNotFoundError(
                    f"adapter weights not found in {adapter_dir}: "
                    "expected adapter_model.safetensors or adapter_model.bin")
            weights_path = bin_path
            disk = torch.load(bin_path, map_location="cpu")

        # -- check 1+2: normalised key match and value equality --------------
        live_sd = self._model.state_dict()
        live_norm = {self._norm_lora_key(k): v.detach().float().cpu()
                     for k, v in live_sd.items() if "lora_" in k}
        disk_norm = {self._norm_lora_key(k): k for k in disk}
        missing = sorted(set(disk_norm) - set(live_norm))
        extra = sorted(set(live_norm) - set(disk_norm))
        max_abs_diff, mismatched = 0.0, []
        for nk, orig in disk_norm.items():
            if nk not in live_norm:
                continue
            d = (live_norm[nk] - disk[orig].float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
            if d > 1e-6:
                mismatched.append({"key": orig, "max_abs_diff": d})

        # -- check 3: PEFT's own loader verdict ------------------------------
        # Re-applying the same state dict is idempotent and yields PEFT's
        # missing/unexpected key report for the base model it is attached to.
        with torch.no_grad():
            load_res = set_peft_model_state_dict(self._model, disk)

        def _pick(res, key):
            if isinstance(res, dict):
                return list(res.get(key, []) or [])
            return list(getattr(res, key, []) or [])

        peft_missing = _pick(load_res, "missing_keys")
        peft_unexpected = _pick(load_res, "unexpected_keys")
        # `missing_keys` would normally also list every *base* parameter that was
        # not in our adapter-only dict; only LoRA keys are meaningful here.
        peft_missing_lora = [k for k in peft_missing if "lora_" in k]

        # -- check 4: module placement + lora_B non-triviality ---------------
        lora_modules = [n for n, _ in self._model.named_modules() if n.endswith("lora_A.default")
                        or n.endswith("lora_A")]
        outside_lm = [n for n in lora_modules if ".language_model." not in n]
        b_nonzero, b_zero = 0, 0
        for nk, orig in disk_norm.items():
            if "lora_B" in nk:
                if disk[orig].float().abs().max().item() > 0:
                    b_nonzero += 1
                else:
                    b_zero += 1
        n_lora_params = sum(v.numel() for v in disk.values())
        live_lora_params = sum(v.numel() for v in live_norm.values())

        weights_verified = (
            len(missing) == 0 and len(extra) == 0 and len(mismatched) == 0
            and not peft_missing_lora and not peft_unexpected
            and len(outside_lm) == 0 and b_zero == 0
            and live_lora_params == n_lora_params
        )

        self._adapter_meta = {
            "adapter_dir": adapter_dir,
            "adapter_config_sha256": sha256_file(os.path.join(adapter_dir, "adapter_config.json")),
            "adapter_weights_sha256": sha256_file(weights_path),
            "r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "target_modules": cfg.get("target_modules"),
            "base_model_name_or_path": cfg.get("base_model_name_or_path"),
            "n_tensors_on_disk": len(disk),
            "n_lora_tensors_live": len(live_norm),
            "n_matched": len(disk_norm) - len(missing),
            "missing_in_model": len(missing),
            "missing_examples": missing[:5],
            "extra_in_model": len(extra),
            "value_mismatched": len(mismatched),
            "value_mismatch_examples": mismatched[:5],
            "max_abs_diff": max_abs_diff,
            "peft_loader_missing_keys": len(peft_missing),
            "peft_loader_missing_lora_keys": len(peft_missing_lora),
            "peft_loader_missing_note": ("missing_keys 里绝大多数是未提供的 4bit 基座权重，"
                                         "只把 lora_ 前缀的缺失视为失败"),
            "peft_loader_unexpected_keys": len(peft_unexpected),
            "n_lora_modules": len(lora_modules),
            "lora_modules_outside_language_model": len(outside_lm),
            "lora_B_nonzero": b_nonzero,
            "lora_B_all_zero": b_zero,
            "lora_params_disk": n_lora_params,
            "lora_params_live": live_lora_params,
            "trainable_params": sum(p.numel() for p in self._model.parameters() if p.requires_grad),
            "model_training_mode": bool(self._model.training),
            "inference_uses_no_grad": True,
            "weights_verified": weights_verified,
        }
        return self._adapter_meta

    # -- inputs --------------------------------------------------------------
    def _image_pad_id(self) -> int:
        tok = self._processor.tokenizer
        return tok.convert_tokens_to_ids("<|image_pad|>")

    def _prepare(self, image_path: str) -> Tuple[Dict, Dict]:
        from PIL import Image
        with Image.open(image_path) as im:
            original_size = list(im.size)
        messages = build_messages(image_path, use_system_prompt=self.use_system_prompt)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        grid = inputs.get("image_grid_thw")
        grid_list = grid.tolist() if grid is not None else None
        ip = self._processor.image_processor
        patch, merge = ip.patch_size, ip.merge_size
        n_patches = int(grid_list[0][0] * grid_list[0][1] * grid_list[0][2]) if grid_list else None
        meta = {
            "image_sha256": sha256_file(image_path),
            "original_size_wh": original_size,
            "max_pixels": self.max_pixels,
            "patch_size": patch,
            "merge_size": merge,
            "image_grid_thw": grid_list,
            "n_patches": n_patches,
            "n_visual_tokens": (n_patches // (merge * merge)) if n_patches else None,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "n_image_pad_tokens": int((inputs["input_ids"] == self._image_pad_id()).sum().item()),
        }
        return inputs, meta

    # -- inference -----------------------------------------------------------
    def recognize_page(self, image_path: str, use_few_shot: bool = True) -> dict:
        self._load()
        import torch

        image_path = resolve_image_path(image_path)
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem_before = gpu_memory_used_mib()

        inputs, meta = self._prepare(image_path)
        prompt_text = self._processor.tokenizer.decode(inputs["input_ids"][0].tolist(),
                                                       skip_special_tokens=False)
        gc = getattr(self._model, "generation_config", None)
        gc_eos = getattr(gc, "eos_token_id", None) if gc is not None else None
        if gc_eos is None:
            gc_eos = self._processor.tokenizer.eos_token_id
        if isinstance(gc_eos, (list, tuple)):
            gc_eos = gc_eos[0] if gc_eos else None
        eos_id = int(gc_eos) if gc_eos is not None else None

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        gen_ids = gen[0].tolist()
        decoded = self._processor.tokenizer.decode(gen[0], skip_special_tokens=True)
        raw_text = decoded[0] if isinstance(decoded, list) else decoded
        mem_after = gpu_memory_used_mib()

        stopped_on_eos = bool(gen_ids) and eos_id is not None and gen_ids[-1] == eos_id
        hit_cap = len(gen_ids) >= self.max_new_tokens
        finish_reason = "eos" if stopped_on_eos else ("length" if hit_cap else "other")

        markdown, wrap_info = normalize_output(raw_text)
        pp = postprocess(markdown)
        rep = detect_repetition(markdown)

        model_tag = f"{os.path.basename(os.path.normpath(self.base_model))}" + (
            f"+{os.path.basename(os.path.normpath(self.adapter_dir))}" if self.adapter_dir else ""
        )
        self.last_run = {
            "protocol_version": PROTOCOL_VERSION,
            "engine": "ocr.engine/OCREngine",
            "base_model": self.base_model,
            "load_in_4bit": self.load_in_4bit,
            "adapter": self._adapter_meta or None,
            "image_path": image_path,
            "prompt": {
                "system_prompt_used": self.use_system_prompt,
                "instruction": INSTRUCTION,
                "prompt_sha256": sha256_text(prompt_text),
                "prompt_tokens": meta["prompt_tokens"],
                "n_image_pad_tokens": meta["n_image_pad_tokens"],
            },
            "image": {
                "sha256": meta["image_sha256"],
                "original_size_wh": meta["original_size_wh"],
                "max_pixels": meta["max_pixels"],
                "patch_size": meta["patch_size"],
                "merge_size": meta["merge_size"],
                "image_grid_thw": meta["image_grid_thw"],
                "n_patches": meta["n_patches"],
                "n_visual_tokens": meta["n_visual_tokens"],
                "processed_size_wh": ([meta["image_grid_thw"][0][2] * meta["patch_size"],
                                       meta["image_grid_thw"][0][1] * meta["patch_size"]]
                                      if meta["image_grid_thw"] else None),
            },
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "generated_tokens": len(gen_ids),
                "eos_token_id": eos_id,
                "last_token_id": gen_ids[-1] if gen_ids else None,
                "stopped_on_eos": stopped_on_eos,
                "hit_length_cap": hit_cap,
                "finish_reason": finish_reason,
            },
            "repetition": rep,
            "postprocess": {
                "raw_chars": wrap_info["raw_chars"],
                "wrapper_stripped": wrap_info["wrapper_stripped"],
                "display_math_converted": wrap_info["display_converted"],
                "inline_math_converted": wrap_info["inline_converted"],
                "normalized_chars": wrap_info["normalized_chars"],
                "markdown_chars": len(pp["markdown"]),
                "changed": wrap_info["changed"],
            },
            "perf": {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated()
                                              if torch.cuda.is_available() else None),
                "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved()
                                             if torch.cuda.is_available() else None),
                "gpu_memory_used_before_mib": mem_before,
                "gpu_memory_used_after_mib": mem_after,
                "gpu_memory_note": ("nvidia-smi memory.used 为整卡占用（含其它进程），"
                                    "Windows WDDM 下不等于本进程独占显存"),
            },
            "structure_ok": pp["structure_ok"],
            "latex_issues": pp["latex_issues"],
            "stats": pp["stats"],
        }
        result = make_result(
            markdown=pp["markdown"],
            structure_ok=pp["structure_ok"],
            latex_issues=pp["latex_issues"],
            tables=pp["stats"]["tables"],
            formulas=pp["stats"]["formulas"],
            model=model_tag,
        )
        result["raw_output"] = raw_text
        return result

    def recognize_batch(self, image_paths, use_few_shot: bool = True):
        return [self.recognize_page(p, use_few_shot) for p in image_paths]

    def describe(self) -> Dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "base_model": self.base_model,
            "adapter": self._adapter_meta or None,
            "max_new_tokens": self.max_new_tokens,
            "max_pixels": self.max_pixels,
            "use_system_prompt": self.use_system_prompt,
            "load_in_4bit": self.load_in_4bit,
            "instruction_sha256": sha256_text(INSTRUCTION),
        }


# Convenience singleton for the default config.
_default_engine: Optional[OCREngine] = None


def recognize_page(image_path: str, adapter_dir: Optional[str] = None, **kwargs):
    """One-shot helper using a module-level default engine.

    The default engine is reused only when no configuration override is given;
    otherwise a temporary engine is built so a later call cannot silently inherit
    an earlier call's ``max_new_tokens`` / ``max_pixels`` / system-prompt choice.
    """
    global _default_engine
    if _default_engine is None or adapter_dir is not None or kwargs:
        eng = OCREngine(adapter_dir=adapter_dir, **kwargs)
        if adapter_dir is None and not kwargs:
            _default_engine = eng
        return eng.recognize_page(image_path)
    return _default_engine.recognize_page(image_path)
