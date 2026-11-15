FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UPLOAD_DIR=/app/data/uploads

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/uploads /app/data/chroma /app/data/memory /app/data/assets /app/logs \
    && chown -R app:app /app

ARG UV_EXTRAS=""
COPY --chown=app:app pyproject.toml uv.lock ./
USER app
# Install third-party dependencies first; the local package needs its source and README.
RUN uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}
COPY --chown=app:app . .
RUN uv sync --frozen --no-dev ${UV_EXTRAS}

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "maitian_agent.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
