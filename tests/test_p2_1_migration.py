"""Regression checks for the LangChain 1.x migration boundary."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_langchain_dependencies_are_locked_to_compatible_1x_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    rag_dependencies = pyproject["project"]["optional-dependencies"]["rag"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}

    assert any(dep.startswith("langchain>=1") for dep in dependencies)
    assert any(dep.startswith("langchain-core>=1") for dep in dependencies)
    assert any(dep.startswith("langchain-openai>=1") for dep in dependencies)
    assert any(dep.startswith("langchain-chroma>=1") for dep in dependencies)
    assert any(dep.startswith("langchain-huggingface>=1") for dep in rag_dependencies)
    assert "langchain-community" not in dependencies
    assert "langchain-community" not in packages
    assert "langchain-classic" not in packages
    rag_locked = packages["maitian-agent"]["optional-dependencies"]["rag"]
    assert "langchain-huggingface" in {dependency["name"] for dependency in rag_locked}


def test_legacy_memory_and_community_embedding_imports_are_removed():
    sources = [
        (ROOT / "maitian_agent/memory/conversation_memory.py").read_text(),
        (ROOT / "maitian_agent/agents/base.py").read_text(),
        (ROOT / "maitian_agent/rag/embeddings.py").read_text(),
    ]
    source = "\n".join(sources)

    assert "langchain.memory" not in source
    assert "langchain_core.memory" not in source
    assert "BaseMemory" not in source
    assert "langchain_community.embeddings" not in source
    assert "langchain_huggingface" in source


def test_default_dependency_set_does_not_install_local_embedding_stack():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = "\n".join(pyproject["project"]["dependencies"])
    rag_dependencies = "\n".join(pyproject["project"]["optional-dependencies"]["rag"])

    assert "langchain-huggingface" not in dependencies
    assert "sentence-transformers" not in dependencies
    assert "torch" not in dependencies
    assert "langchain-huggingface" in rag_dependencies
