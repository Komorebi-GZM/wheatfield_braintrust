# 麦田智囊 - 乡村教育Agent系统

基于LangChain与自进化RAG的乡村教育Agent系统，为乡村教师提供专属AI教研搭档。

## 核心功能

- 📚 **乡村专属极速备课**：10秒生成乡土化教案
- 📝 **老教师经验一键传承**：手写教案OCR识别→结构化→入库→可检索
- 🎯 **课堂实时伴教**：3秒极速出题、经典题检索
- 🎬 **具象化素材推荐**：自动匹配科普视频/3D模型/动画
- 📋 **教研纪要生成**：录音转写→结构化报告→成果入库

## 技术架构

### 四层架构

- **感知层**：ASR语音识别 + OCR手写体识别
- **规划层**：LangChain Router 意图路由器
- **执行层**：5大核心Agent执行链 + RAG检索引擎
- **记忆层**：短期对话记忆 + 长期教师专属画像

### 五大Agent链

1. QuickLessonPrepAgent - 极速备课
2. WisdomTransferAgent - 智慧传承
3. ClassroomCompanionAgent - 课堂伴教
4. MaterialAgent - 素材推荐
5. MeetingNotesAgent - 教研纪要

## 快速开始

### 安装依赖

```bash
uv sync
uv run maitian-agent --help
```

### 配置环境变量

复制 `.env.example` 为 `.env`，按需填写模型配置（密钥只放 `.env`，不要提交）：

```bash
cp .env.example .env
```

### 知识库向量维度迁移

默认嵌入模型已从 `BAAI/bge-large-zh`（1024 维）改为 `BAAI/bge-small-zh-v1.5`（512 维）。

- 已有 `./data/chroma` 索引目录是旧模型（1024 维）生成的，**不能**与新模型（512 维）混用；
- 打开知识库时会校验已存向量的维度，若与当前 embedding 维度不一致会直接报错（提示已存维度与当前维度）；
- 处理办法（二选一）：
  1. 把 `EMBEDDING_MODEL` 设回原来的模型（例如 `BAAI/bge-large-zh`），继续使用旧索引；
  2. 备份或删除原索引目录（如 `./data/chroma`）后，用当前模型重建索引。

### 启动应用

```bash
# 在两个终端分别启动 Web 界面与 API
# 启动Web界面
uv run streamlit run maitian_agent/frontend/streamlit_app.py --server.address 127.0.0.1

# 启动API服务
uv run uvicorn maitian_agent.api.routes:app --host 127.0.0.1 --port 8000
```

### Docker 一键启动

在仓库根目录使用 `compose.yaml` 运行（Docker Compose 2.24+，没有 `.env` 也能启动）：

```bash
docker compose up --build -d
# 停止并清理容器
docker compose down
```

访问 API http://127.0.0.1:8000/health 和 Streamlit http://127.0.0.1:8501 。
默认端口只绑定本机 `127.0.0.1`（容器内服务仍监听 `0.0.0.0`）。若要对外暴露，必须自己去掉 `127.0.0.1` 绑定，并且加上认证（例如放在反向代理之后并启用鉴权）。不要只改端口映射就直接暴露到公网。
无 Key 时健康检查仍返回 200；生成能力需要补齐 `LLM_*` 配置。
两个服务共用镜像，前端通过 `MAITIAN_API_BASE=http://api:8000` 访问 API。
两者挂载 `./data:/app/data`，上传目录固定为 `/app/data/uploads`。
一次性 `data-init` 服务创建子目录并将宿主数据目录所有权设为 UID/GID 1000，应用以非 root 运行。

环境变量见 `.env.example`：支持 LLM、嵌入、ASR（`ASR_BASE_URL` / `ASR_MODEL` / `ASR_API_KEY`）、OCR、CORS 和上传大小。
默认镜像不含 torch、EasyOCR 或本地嵌入依赖，`OCR_BACKEND=none`。
需要时选择对应 extras（首次使用还可能下载模型）：

```bash
# 本地安装，可只选一个 extra
uv sync --extra rag --extra ocr-easyocr
# 单独构建镜像
docker build --build-arg UV_EXTRAS="--extra rag --extra ocr-easyocr" -t wheatfield:p1-5 .
# Compose 构建并启动可选版本
UV_EXTRAS="--extra rag --extra ocr-easyocr" docker compose up --build -d
```

根目录 `Dockerfile` 与 `compose.yaml` 使用锁定依赖，CI 的 `docker build -t maitian-agent:ci .` 会自动构建根目录 Dockerfile。
需要 Docker 组权限时可用 `sg docker -c "docker compose up --build -d"`。

## 项目结构

```
maitian_agent/
├── agents/          # Agent模块
│   ├── base.py      # Agent基类
│   ├── quick_lesson_prep.py
│   ├── wisdom_transfer.py
│   ├── classroom_companion.py
│   ├── material_agent.py
│   ├── meeting_notes.py
│   └── router.py
├── rag/             # RAG模块
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── knowledge_base.py
├── memory/          # 记忆模块
│   ├── conversation_memory.py
│   └── teacher_profile.py
├── tools/           # 工具模块
│   ├── ocr.py
│   ├── asr.py
│   └── file.py
├── api/             # API模块
│   └── routes.py
├── config/          # 配置模块
│   └── settings.py
└── frontend/        # 前端模块
    └── streamlit_app.py
```

## 环境要求

- Python 3.12+
- OpenAI API Key 或兼容API

## License

© 2026 麦田智囊团队
