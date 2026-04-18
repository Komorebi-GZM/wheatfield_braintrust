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
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env`，填入API密钥：

```bash
cp .env.example .env
```

### 启动应用

```bash
# 启动Web界面
streamlit run maitian_agent/frontend/streamlit_app.py

# 或启动API服务
uvicorn maitian_agent.api.routes:app --host 0.0.0.0 --port 8000
```

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

- Python 3.10+
- OpenAI API Key 或兼容API

## 文档

详细技术文档请参考 `开发文档/` 目录。

## License

© 2026 麦田智囊团队
