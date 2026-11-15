"""Optional backends stay lazy and explain how to install dependencies."""

import sys
from unittest.mock import Mock

import pytest

from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.tools.ocr import EasyOCRProcessor, PaddleOCRProcessor


@pytest.mark.parametrize(
    ("module", "factory", "attribute", "hint"),
    [
        ("easyocr", EasyOCRProcessor, "reader", "maitian-agent[ocr-easyocr]"),
        ("paddleocr", PaddleOCRProcessor, "ocr", "PaddleOCR 未安装"),
    ],
)
def test_missing_ocr_dependency(monkeypatch, module, factory, attribute, hint):
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError) as error:
        getattr(factory(), attribute)

    assert hint in str(error.value)
    assert isinstance(error.value.__cause__, ImportError)


def test_missing_rag_dependency(monkeypatch):
    from maitian_agent.rag.embeddings import BGEEmbeddings

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match=r"maitian-agent\[rag\]"):
        _ = BGEEmbeddings().embeddings


def test_wisdom_transfer_reuses_lazy_reader():
    agent = WisdomTransferAgent(llm=Mock())
    assert isinstance(agent.ocr, EasyOCRProcessor)
    assert agent.ocr._reader is None

    reader = Mock()
    reader.readtext.return_value = [(None, "第一行", 0.9), (None, "第二行", 0.8)]
    agent.ocr._reader = reader

    assert agent.recognize_handwriting("lesson.png") == "第一行\n第二行"
    reader.readtext.assert_called_once_with("lesson.png", detail=1)


def test_wisdom_transfer_supports_injected_reader():
    reader = Mock()
    reader.readtext.return_value = [(None, "教案", 0.9)]
    agent = WisdomTransferAgent(llm=Mock(), ocr_reader=reader)

    assert agent.recognize_handwriting("lesson.png") == "教案"
    reader.readtext.assert_called_once_with("lesson.png", detail=1)
