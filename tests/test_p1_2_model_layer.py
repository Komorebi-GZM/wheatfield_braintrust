"""P1-2 配置与厂商无关模型层测试（全部离线，不需要任何 Key）。"""
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import maitian_agent.llm as llm_module
from maitian_agent.config.settings import get_settings

_ENV_PREFIXES = ("LLM_", "OPENAI_", "EMBEDDING_")


def _is_model_env(key: str) -> bool:
    upper = key.upper()
    return upper.startswith(_ENV_PREFIXES) or upper in ("MODEL_NAME",)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """清除进程里可能存在的 LLM/OPENAI/EMBEDDING 环境变量，并重置缓存的 settings。"""
    for key in list(os.environ):
        if _is_model_env(key):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    llm_module.reset_chat_model()
    yield
    get_settings.cache_clear()
    llm_module.reset_chat_model()


def test_settings_defaults_without_env():
    s = get_settings()
    assert s.llm_temperature == 0.7
    assert s.llm_timeout_seconds == 60
    assert s.llm_max_retries == 2
    assert s.embedding_backend == "local"
    assert s.resolved_embedding_model == "BAAI/bge-small-zh-v1.5"
    assert s.llm_configured is False


def test_get_chat_model_not_configured():
    with pytest.raises(llm_module.LLMNotConfiguredError) as exc_info:
        llm_module.get_chat_model()
    assert exc_info.value.code == "LLM_NOT_CONFIGURED"


def test_get_chat_model_configured(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-placeholder")
    get_settings.cache_clear()

    model = llm_module.get_chat_model()
    assert isinstance(model, BaseChatModel)
    # 只验证构造，不触发任何网络请求
    assert getattr(model, "model_name", None) == "fake-model"


def test_injected_fake_model_overrides_factory():
    fake = FakeListChatModel(responses=["任意输出"])
    llm_module.set_chat_model(fake)
    # 即便未配置也应直接返回覆盖模型
    assert llm_module.get_chat_model() is fake


def test_production_agent_uses_injected_fake_model():
    from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent

    fake = FakeListChatModel(responses=["这是生成的教案内容"])
    agent = QuickLessonPrepAgent(llm=fake)
    output = agent.run(subject="语文", grade="三年级", topic="春天", rural_context="乡村")
    assert "这是生成的教案内容" in output


def test_demo_agent_uses_injected_fake_model():
    from demo.agents.quick_lesson_prep import QuickLessonPrepAgent as DemoAgent

    fake = FakeListChatModel(responses=["demo 教案内容"])
    agent = DemoAgent(llm=fake)
    output = agent.run(subject="数学", grade="四年级", topic="分数", rural_context="乡村")
    assert "demo 教案内容" in output


def test_health_returns_200_and_not_configured():
    from maitian_agent.api.routes import app

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_configured"] is False
    assert body["status"] == "healthy"


def test_generation_endpoint_returns_llm_not_configured():
    from maitian_agent.api.routes import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/lesson-prep",
            json={"subject": "语文", "grade": "三年级", "topic": "春天"},
        )

    assert resp.status_code >= 400
    detail = resp.json()["detail"]
    assert detail["code"] == "LLM_NOT_CONFIGURED"


def test_import_app_does_not_load_torch_or_easyocr():
    """在干净子进程中导入应用，断言未加载 torch/easyocr。"""
    code = (
        "import sys\n"
        "import maitian_agent.api.routes  # noqa: F401\n"
        "print(int('torch' in sys.modules), int('easyocr' in sys.modules))\n"
    )
    clean_env = {k: v for k, v in os.environ.items() if not _is_model_env(k)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        env=clean_env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 0", (
        f"torch/easyocr 被意外加载: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_deterministic_fake_embedding_available():
    """确认测试可用的假嵌入（供后续 RAG 测试复用）。"""
    emb = DeterministicFakeEmbedding(size=8)
    vec = emb.embed_query("你好")
    assert len(vec) == 8
