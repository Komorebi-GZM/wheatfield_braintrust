"""Offline regressions for memory, blocking generation and lazy BGE setup."""
import asyncio
import json
import sys
import threading
import time
from io import BytesIO
from unittest.mock import Mock
from types import ModuleType

import httpx
from fastapi.testclient import TestClient
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from PIL import Image
from pydantic import PrivateAttr

from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.api import routes
from maitian_agent.config.settings import get_settings
from maitian_agent.memory.conversation_memory import ConversationMemory
from maitian_agent.rag.embeddings import BGEEmbeddings


def test_memory_without_history_save_reload_and_clear(tmp_path):
    memory = ConversationMemory(persist_directory=str(tmp_path))
    assert memory.load_memory_variables()["history"] == []
    memory.save_context({"human_input": "问题"}, {"response": "回答"})
    restored = ConversationMemory(persist_directory=str(tmp_path))
    assert [(m.type, m.content) for m in restored.load_memory_variables()["history"]] == [
        ("human", "问题"), ("ai", "回答")
    ]
    restored.clear()
    assert restored.load_memory_variables()["history"] == []
    assert ConversationMemory(persist_directory=str(tmp_path)).get_conversation_history() == []


def test_memory_loads_json_and_respects_window(tmp_path):
    (tmp_path / "chat_history_default.json").write_text(json.dumps([
        {"type": "human", "content": "旧问题"}, {"type": "ai", "content": "旧回答"},
        {"type": "human", "content": "新问题"}, {"type": "ai", "content": "新回答"},
    ]), encoding="utf-8")
    memory = ConversationMemory(conversation_window=1, persist_directory=str(tmp_path))
    assert [(m.type, m.content) for m in memory.load_memory_variables()["history"]] == [
        ("human", "新问题"), ("ai", "新回答")
    ]
    assert len(memory.get_conversation_history()) == 4


def test_bge_maps_passage_to_embed_instruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", ModuleType("sentence_transformers"))
    module = ModuleType("langchain_community.embeddings")
    calls = []

    class StrictBGE:
        def __init__(self, *, model_name, model_kwargs, encode_kwargs,
                     query_instruction, embed_instruction):
            calls.append((query_instruction, embed_instruction))

    module.HuggingFaceBgeEmbeddings = StrictBGE
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", module)
    embedding = BGEEmbeddings(query_instruction="查询", passage_instruction="文档")
    assert isinstance(embedding.embeddings, StrictBGE)
    assert embedding.embeddings is embedding.embeddings
    assert calls == [("查询", "文档")]


class SlowFakeChatModel(FakeListChatModel):
    _started: threading.Event = PrivateAttr(default_factory=threading.Event)

    def _call(self, *args, **kwargs):
        self._started.set()
        time.sleep(1.5)
        return super()._call(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["lesson-prep", "wisdom-transfer"])
async def test_health_responsive_during_slow_generation(monkeypatch, tmp_path, endpoint):
    for key, value in {"LLM_BASE_URL": "http://fake.invalid/v1", "LLM_MODEL": "fake",
                       "LLM_API_KEY": "placeholder", "UPLOAD_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    model = SlowFakeChatModel(responses=["生成结果"])

    class Reader:
        def readtext(self, *args, **kwargs):
            return [(None, "教案", 1.0)]

    app = routes.create_app()
    try:
        async with app.router.lifespan_context(app):
            monkeypatch.setattr(routes, "agents", {
                "quick_lesson_prep": QuickLessonPrepAgent(llm=model),
                "wisdom_transfer": WisdomTransferAgent(llm=model, ocr_reader=Reader()),
            })
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test") as client:
                if endpoint == "lesson-prep":
                    kwargs = {"json": {"subject": "语文", "grade": "三年级", "topic": "春天"}}
                else:
                    image = BytesIO()
                    Image.new("RGB", (1, 1)).save(image, format="PNG")
                    kwargs = {"files": {"file": ("lesson.png", image.getvalue(), "image/png")}}
                task = asyncio.create_task(client.post(f"/api/{endpoint}", **kwargs))
                try:
                    async with asyncio.timeout(3):
                        while not model._started.is_set():
                            await asyncio.sleep(0.005)
                    assert not task.done(), "生成请求阻塞了事件循环，直到模型完成才恢复"
                    start = time.monotonic()
                    response = await client.get("/health")
                    assert time.monotonic() - start < 0.5
                    assert response.status_code == 200
                    assert not task.done()
                finally:
                    generated = await task
                assert generated.status_code == 200
                assert "生成结果" in generated.json()["result"]
    finally:
        get_settings.cache_clear()



@pytest.mark.parametrize(
    "path,payload,agent_key,method",
    [
        ("/api/lesson-prep", {"subject": "语文", "grade": "三年级", "topic": "春天"},
         "quick_lesson_prep", "run"),
        ("/api/quiz", {"subject": "语文", "grade": "三年级", "topic": "春天"},
         "classroom_companion", "generate_quiz"),
        ("/api/material", {"subject": "语文", "grade": "三年级", "topic": "春天"},
         "material", "recommend_materials"),
        ("/api/meeting-notes", {"transcript": "教研记录"},
         "meeting_notes", "process_meeting_notes"),
        ("/api/route", {"user_input": "帮我备课"}, "router", "run"),
    ],
)
def test_agent_error_returns_generic_500(monkeypatch, tmp_path, path, payload, agent_key, method):
    """Agent 抛错时 API 返回通用中文 500，响应体不含异常细节。"""
    for key, value in {
        "LLM_BASE_URL": "http://fake.invalid/v1",
        "LLM_MODEL": "fake",
        "LLM_API_KEY": "placeholder",
        "UPLOAD_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        client = TestClient(routes.create_app())
        failing = Mock()
        getattr(failing, method).side_effect = RuntimeError("SECRET_TRACE_xyz")
        monkeypatch.setitem(routes.agents, agent_key, failing)
        response = client.post(path, json=payload)
        client.close()
    finally:
        get_settings.cache_clear()
    assert response.status_code == 500
    assert response.json()["detail"] == "服务内部错误，请稍后重试"
    assert "SECRET_TRACE_xyz" not in response.text
    assert "Traceback" not in response.text


def test_wisdom_transfer_agent_error_returns_generic_500(monkeypatch, tmp_path):
    """上传链路中 Agent 抛错同样不得把异常文本回给客户端。"""
    for key, value in {
        "LLM_BASE_URL": "http://fake.invalid/v1",
        "LLM_MODEL": "fake",
        "LLM_API_KEY": "placeholder",
        "UPLOAD_DIR": str(tmp_path / "uploads"),
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        client = TestClient(routes.create_app())
        failing = Mock()
        failing.run.side_effect = RuntimeError("SECRET_TRACE_xyz")
        monkeypatch.setitem(routes.agents, "wisdom_transfer", failing)
        image = BytesIO()
        Image.new("RGB", (1, 1)).save(image, format="PNG")
        response = client.post(
            "/api/wisdom-transfer",
            files={"file": ("lesson.png", image.getvalue(), "image/png")},
        )
        client.close()
    finally:
        get_settings.cache_clear()
    assert response.status_code == 500
    assert response.json()["detail"] == "服务内部错误，请稍后重试"
    assert "SECRET_TRACE_xyz" not in response.text


def test_quick_lesson_prep_raises_instead_of_returning_error():
    class Boom(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("SECRET_TRACE_xyz")

    agent = QuickLessonPrepAgent(llm=Boom(responses=["unused"]))
    with pytest.raises(RuntimeError, match="SECRET_TRACE_xyz"):
        agent.run(subject="语文", grade="三年级", topic="春天", rural_context="")
