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

_ENV_PREFIXES = ("LLM_", "OPENAI_", "EMBEDDING_", "ASR_")


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
    from maitian_agent.agents.results import LessonPrepInput

    output = agent.run(
        LessonPrepInput(subject="语文", grade="三年级", topic="春天", rural_context="乡村")
    )
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
    """在干净子进程中导入应用，断言未加载 torch/easyocr/sentence_transformers。"""
    code = (
        "import sys\n"
        "import maitian_agent.api.routes  # noqa: F401\n"
        "print(int('torch' in sys.modules), int('easyocr' in sys.modules),"
        " int('sentence_transformers' in sys.modules))\n"
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
    assert result.stdout.strip() == "0 0 0", (
        f"torch/easyocr/sentence_transformers 被意外加载: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_deterministic_fake_embedding_available():
    """确认测试可用的假嵌入（供后续 RAG 测试复用）。"""
    emb = DeterministicFakeEmbedding(size=8)
    vec = emb.embed_query("你好")
    assert len(vec) == 8


# ---------------------------------------------------------------------------
# P0：openai_compatible 嵌入不得回退到官方端点
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_env",
    [
        {"EMBEDDING_API_KEY": "sk-emb-placeholder"},
        {"LLM_API_KEY": "sk-llm-placeholder"},
        {"OPENAI_API_KEY": "sk-openai-placeholder"},
        {
            "EMBEDDING_API_KEY": "sk-emb-placeholder",
            "LLM_API_KEY": "sk-llm-placeholder",
            "OPENAI_API_KEY": "sk-openai-placeholder",
        },
        # 有 key、有 model、无 base_url：专门钉死官方端点回退
        {
            "EMBEDDING_API_KEY": "sk-emb-placeholder",
            "EMBEDDING_MODEL": "text-embedding-3-small",
        },
        # 环境里有 OPENAI_BASE_URL 也不得被使用
        {
            "OPENAI_API_KEY": "sk-openai-placeholder",
            "OPENAI_BASE_URL": "https://official.invalid/v1",
        },
    ],
)
def test_openai_compatible_missing_base_url_raises_without_constructing(monkeypatch, extra_env):
    from unittest.mock import Mock

    from maitian_agent.rag.embeddings import EmbeddingNotConfiguredError, get_embeddings

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai_compatible")
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", ctor)

    with pytest.raises(EmbeddingNotConfiguredError):
        get_embeddings()

    # 构造器一次都不能被调用（绝不能把 None 传给 SDK 打到官方端点）
    ctor.assert_not_called()


def test_openai_compatible_configured_uses_explicit_values(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.rag.embeddings import get_embeddings

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-embed")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb-placeholder")
    # 环境里同时有 OPENAI_* 也不得被使用
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-use")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://official.invalid/v1")
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", ctor)

    embeddings = get_embeddings()

    ctor.assert_called_once()
    kwargs = ctor.call_args.kwargs
    assert kwargs["base_url"] == "http://127.0.0.1:9/v1"
    assert kwargs["model"] == "fake-embed"
    assert kwargs["api_key"].get_secret_value() == "sk-emb-placeholder"
    assert embeddings is ctor.return_value


def test_openai_compatible_falls_back_to_llm_credentials(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.rag.embeddings import get_embeddings

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_MODEL", "llm-embed-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm-placeholder")
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", ctor)

    get_embeddings()

    ctor.assert_called_once()
    kwargs = ctor.call_args.kwargs
    assert kwargs["base_url"] == "http://127.0.0.1:9/v1"
    assert kwargs["model"] == "llm-embed-model"
    assert kwargs["api_key"].get_secret_value() == "sk-llm-placeholder"


# ---------------------------------------------------------------------------
# P1-3：local 默认 bge-small；openai_compatible 未设 model 不得用 BGE 名
# ---------------------------------------------------------------------------


def test_local_backend_defaults_to_bge_small():
    s = get_settings()
    assert s.embedding_backend == "local"
    assert s.resolved_embedding_model == "BAAI/bge-small-zh-v1.5"


def test_openai_compatible_without_model_does_not_use_bge(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.rag.embeddings import EmbeddingNotConfiguredError, get_embeddings

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-emb-placeholder")
    # 不设 EMBEDDING_MODEL，也不设 LLM_MODEL
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", ctor)

    with pytest.raises(EmbeddingNotConfiguredError):
        get_embeddings()
    ctor.assert_not_called()


# ---------------------------------------------------------------------------
# P1-1：ASR 独立配置
# ---------------------------------------------------------------------------


def test_whisper_asr_not_configured_raises(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.tools.asr import ASRNotConfiguredError, WhisperASR

    ctor = Mock()
    monkeypatch.setattr("openai.OpenAI", ctor)

    with pytest.raises(ASRNotConfiguredError):
        WhisperASR()
    ctor.assert_not_called()


def test_whisper_asr_only_openai_key_raises(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.tools.asr import ASRNotConfiguredError, WhisperASR

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-placeholder")
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("openai.OpenAI", ctor)

    with pytest.raises(ASRNotConfiguredError):
        WhisperASR()
    ctor.assert_not_called()


def test_whisper_asr_configured_uses_explicit_values(monkeypatch):
    from unittest.mock import Mock

    from maitian_agent.tools.asr import WhisperASR

    monkeypatch.setenv("ASR_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ASR_MODEL", "whisper-1")
    monkeypatch.setenv("ASR_API_KEY", "sk-asr-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-use")
    get_settings.cache_clear()

    ctor = Mock()
    monkeypatch.setattr("openai.OpenAI", ctor)

    asr = WhisperASR()
    _ = asr.client

    ctor.assert_called_once()
    kwargs = ctor.call_args.kwargs
    assert kwargs["api_key"] == "sk-asr-placeholder"
    assert kwargs["base_url"] == "http://127.0.0.1:9/v1"


# ---------------------------------------------------------------------------
# P1-2：知识库向量维度校验
# ---------------------------------------------------------------------------


def test_dimension_mismatch_raises(tmp_path):
    from maitian_agent.rag.vectorstore import (
        ChromaVectorStore,
        EmbeddingDimensionMismatchError,
    )

    # 用 1024 维写入
    store = ChromaVectorStore(
        collection_name="dimtest",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=1024),
    )
    store.add_texts(["第一段文本"])

    # 以 512 维重新打开同一 collection，必须报错
    reopened = ChromaVectorStore(
        collection_name="dimtest",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=512),
    )
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        _ = reopened.vectorstore

    message = str(exc_info.value)
    assert "1024" in message and "512" in message
    assert "EMBEDDING_MODEL" in message
    assert "重建" in message


def test_new_collection_writes_dimension_and_same_dim_reopens(tmp_path):
    from maitian_agent.rag.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(
        collection_name="metatest",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=512),
    )
    store.add_texts(["一段文本"])

    collection = store.vectorstore._collection
    assert (collection.metadata or {}).get("embedding_dimension") == 512

    # 同维度再次打开应成功
    reopened = ChromaVectorStore(
        collection_name="metatest",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=512),
    )
    assert reopened.vectorstore is not None


def test_legacy_collection_without_metadata_probes_dimension(tmp_path):
    from langchain_chroma import Chroma

    from maitian_agent.rag.vectorstore import (
        ChromaVectorStore,
        EmbeddingDimensionMismatchError,
    )

    # 直接构造 Chroma（绕过本模块），模拟没有维度元数据的遗留 collection
    raw = Chroma(
        collection_name="legacy",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=1024),
    )
    raw.add_texts(["遗留向量"])
    assert (raw._collection.metadata or {}).get("embedding_dimension") is None

    # 以 512 维打开，probe 到 1024 维后必须报错
    reopened = ChromaVectorStore(
        collection_name="legacy",
        persist_directory=str(tmp_path),
        embedding_function=DeterministicFakeEmbedding(size=512),
    )
    with pytest.raises(EmbeddingDimensionMismatchError):
        _ = reopened.vectorstore
