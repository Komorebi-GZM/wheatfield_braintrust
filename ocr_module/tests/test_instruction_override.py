"""Prompt-override plumbing used by the prompt-factor experiments.

The frozen protocol must stay the default: `build_messages()` and
`OCREngine()` keep emitting the training instruction unless a caller passes an
explicit override, and `describe()` must report the effective instruction so an
output sidecar can never claim the wrong prompt.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engine import OCREngine  # noqa: E402
from ocr.prompts import INSTRUCTION, build_messages  # noqa: E402

CUSTOM = "转写图中的手写内容。只输出图中实际存在的标题与条目，不要补充图中没有的内容。"


def _text_of(msgs):
    texts = [b["text"] for b in msgs[-1]["content"] if b["type"] == "text"]
    assert len(texts) == 1
    return texts[0]


def test_default_messages_use_frozen_instruction():
    msgs = build_messages("page.jpg")
    assert _text_of(msgs) == "\n" + INSTRUCTION
    assert msgs[0]["role"] == "user"  # no system message by default


def test_override_replaces_only_the_instruction_text():
    msgs = build_messages("page.jpg", instruction=CUSTOM)
    assert _text_of(msgs) == "\n" + CUSTOM
    assert INSTRUCTION not in _text_of(msgs)
    # the image block and the leading newline must be untouched
    assert msgs[-1]["content"][0] == {"type": "image", "url": "page.jpg"}
    assert _text_of(msgs).startswith("\n")


def test_engine_defaults_to_frozen_instruction():
    desc = OCREngine().describe()
    assert desc["instruction"] == INSTRUCTION
    assert desc["instruction_sha256"] == hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()


def test_engine_reports_override():
    desc = OCREngine(instruction=CUSTOM).describe()
    assert desc["instruction"] == CUSTOM
    assert desc["instruction_sha256"] == hashlib.sha256(CUSTOM.encode("utf-8")).hexdigest()
