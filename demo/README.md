# 麦田智囊 Demo

乡村教育Agent系统的交互式演示版本，基于Streamlit构建。

## 功能演示

### 1. 乡村专属极速备课 ✅ 已完成
- **输入**：学科、年级、课题、乡村特色情境
- **输出**：详细的乡土化教案
- **特点**：10秒生成，结合乡村实际案例

### 2. 老教师经验一键传承 ✅ 已完成
- **输入**：手写教案照片
- **输出**：OCR识别结果 + 结构化教案
- **特点**：EasyOCR识别，大模型结构化处理

### 3. 课堂实时伴教 🚧 开发中
- 练习题生成
- 经典题检索

### 4. 具象化素材推荐 🚧 开发中
- 教学视频推荐
- 3D模型/动画素材

### 5. 教研纪要生成 🚧 开发中
- 会议记录转写
- 结构化报告生成

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑.env，填入API密钥
```

### 启动演示

```bash
streamlit run demo/app.py
```

访问 http://localhost:8501 查看演示。

## 技术栈

- **前端**：Streamlit
- **编排**：LangChain LCEL
- **大模型**：DeepSeek-V2 API
- **OCR**：EasyOCR
- **向量库**：Chroma

## 目录结构

```
demo/
├── app.py                    # Streamlit主应用
├── agents/
│   ├── quick_lesson_prep.py # 极速备课Agent
│   └── wisdom_transfer.py    # 智慧传承Agent
├── .env.example              # 环境变量示例
└── README.md                 # 本文件
```

## 完整项目

完整项目框架请参考项目根目录的 `maitian_agent/` 模块，包含：
- 5大核心Agent链
- 自进化RAG知识库
- 对话记忆+教师画像
- FastAPI REST接口
- Docker容器化部署

## License

© 2026 麦田智囊团队
