"""Decoding-knob plumbing for the decode-factor experiments.

Defaults must stay byte-identical to the frozen protocol: `generate` receives no
`repetition_penalty` / `min_new_tokens` argument at all unless a caller opts in.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engine import OCREngine  # noqa: E402


def test_defaults_omit_both_knobs():
    kw = OCREngine()._generation_kwargs()
    assert "repetition_penalty" not in kw
    assert "min_new_tokens" not in kw
    assert kw["do_sample"] is False
    assert kw["return_dict_in_generate"] is True


def test_explicit_knobs_are_forwarded():
    kw = OCREngine(repetition_penalty=1.1, min_new_tokens=256)._generation_kwargs()
    assert kw["repetition_penalty"] == 1.1
    assert kw["min_new_tokens"] == 256


def test_one_point_zero_is_treated_as_unset():
    engine = OCREngine(repetition_penalty=1.0, min_new_tokens=0)
    kw = engine._generation_kwargs()
    assert "repetition_penalty" not in kw and "min_new_tokens" not in kw
    assert engine.describe()["repetition_penalty"] == 1.0
    assert engine.describe()["min_new_tokens"] == 0


def test_describe_reports_knobs():
    desc = OCREngine(repetition_penalty=1.15, min_new_tokens=128).describe()
    assert desc["repetition_penalty"] == 1.15
    assert desc["min_new_tokens"] == 128
    assert desc["max_new_tokens"] > 0
