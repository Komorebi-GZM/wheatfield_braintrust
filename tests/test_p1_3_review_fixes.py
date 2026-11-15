"""Review follow-ups: EasyOCR init lock and UI error wrapping."""
import sys
import threading
import time
from types import ModuleType

from maitian_agent.agent_call import GENERIC_AGENT_ERROR, invoke_agent
from maitian_agent.tools.ocr import EasyOCRProcessor


def test_easyocr_reader_initialized_once_under_concurrency(monkeypatch):
    constructed = []

    class StubReader:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)
            constructed.append(self)

        def readtext(self, *args, **kwargs):
            time.sleep(0.01)
            return [([0, 0, 1, 1], "教案", 0.99)]

    module = ModuleType("easyocr")
    module.Reader = StubReader
    monkeypatch.setitem(sys.modules, "easyocr", module)

    processor = EasyOCRProcessor()
    errors = []

    def worker():
        try:
            processor.recognize("lesson.png", detail=0)
        except Exception as exc:  # pragma: no cover - should not happen
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(constructed) == 1
    assert processor.reader is constructed[0]


def test_invoke_agent_hides_exception_text():
    shown: list[str] = []

    def boom():
        raise RuntimeError("SECRET_TRACE_xyz")

    result = invoke_agent(boom, error_handler=shown.append)
    assert result is None
    assert shown == [GENERIC_AGENT_ERROR]
    assert "SECRET_TRACE_xyz" not in shown[0]
    assert "Traceback" not in shown[0]


def test_invoke_agent_returns_success_value():
    shown: list[str] = []
    result = invoke_agent(lambda: "教案内容", error_handler=shown.append)
    assert result == "教案内容"
    assert shown == []
