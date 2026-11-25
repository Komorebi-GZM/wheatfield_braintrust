"""Decoding-knob plumbing and protocol-v4 freeze.

Protocol v4 sets `deliver_v4` as the engine default (repetition_penalty=1.1,
min_new_tokens unset). The frozen v3 numeric path must stay reachable by passing
repetition_penalty=1.0 explicitly, and the protocol file must not drift from the
engine constants.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr import engine as engine_mod  # noqa: E402
from ocr.engine import OCREngine  # noqa: E402

PROTOCOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "inference_protocol_v4.json")


def _protocol():
    with open(PROTOCOL, encoding="utf-8") as f:
        return json.load(f)


def test_default_engine_uses_v4_deliver_profile():
    eng = OCREngine()
    kw = eng._generation_kwargs()
    assert kw["repetition_penalty"] == 1.1
    assert "min_new_tokens" not in kw  # 0 means "do not pass"
    assert eng.describe()["repetition_penalty"] == 1.1
    assert eng.describe()["min_new_tokens"] == 0


def test_explicit_one_point_zero_reproduces_v3():
    kw = OCREngine(repetition_penalty=1.0)._generation_kwargs()
    assert "repetition_penalty" not in kw
    assert "min_new_tokens" not in kw


def test_compare_profile_knobs_are_forwarded():
    kw = OCREngine(repetition_penalty=1.1, min_new_tokens=256)._generation_kwargs()
    assert kw["repetition_penalty"] == 1.1
    assert kw["min_new_tokens"] == 256


def test_describe_reports_knobs():
    desc = OCREngine(repetition_penalty=1.15, min_new_tokens=128).describe()
    assert desc["repetition_penalty"] == 1.15
    assert desc["min_new_tokens"] == 128


def test_protocol_file_matches_engine_defaults():
    """Guard against silent drift between inference_protocol_v4.json and the code."""
    proto = _protocol()
    defaults = proto["engine_defaults"]
    deliver = proto["profiles"]["deliver_v4"]

    assert engine_mod.DEFAULT_REPETITION_PENALTY == defaults["repetition_penalty"]
    assert engine_mod.DEFAULT_MIN_NEW_TOKENS == defaults["min_new_tokens"]
    assert engine_mod.DEFAULT_MAX_NEW_TOKENS == defaults["max_new_tokens"]
    assert engine_mod.DEFAULT_MAX_PIXELS == defaults["max_pixels"]

    # deliver_v4 is exactly the engine default profile
    assert deliver["repetition_penalty"] == engine_mod.DEFAULT_REPETITION_PENALTY
    assert deliver["min_new_tokens"] == engine_mod.DEFAULT_MIN_NEW_TOKENS
    assert deliver["max_new_tokens"] == engine_mod.DEFAULT_MAX_NEW_TOKENS
    assert deliver["max_pixels"] == engine_mod.DEFAULT_MAX_PIXELS

    # compare_v4 differs only by the min_new_tokens floor
    compare = proto["profiles"]["compare_v4"]
    assert compare["repetition_penalty"] == deliver["repetition_penalty"]
    assert compare["max_new_tokens"] == deliver["max_new_tokens"]
    assert compare["max_pixels"] == deliver["max_pixels"]
    assert compare["min_new_tokens"] > 0


def test_protocol_instruction_matches_frozen_prompt():
    import hashlib

    from ocr.prompts import INSTRUCTION

    proto = _protocol()
    assert proto["prompt"]["instruction"] == INSTRUCTION
    assert proto["prompt"]["instruction_sha256"] == hashlib.sha256(
        INSTRUCTION.encode("utf-8")).hexdigest()
