# 麦田智囊 - Demo版使用说明

## 项目介绍
"麦田智囊"是基于LangChain与自进化RAG的乡村教育Agent系统，致力于为乡村教师提供专属的AI教研搭档。

## 核心功能
1. **乡村专属极速备课**：10秒语音生成乡土化教案
2. **老教师经验一键传承**：手写教案OCR识别→结构化→入库→可检索

## 技术栈
- 前端：Streamlit
- 核心编排：LangChain 0.2.x（LCEL）
- 大模型：DeepSeek-V2 API
- RAG：Chroma + BGE-Large-zh
- OCR：PaddleOCR
- 语音：Whisper API

## 安装步骤

### 1. 安装依赖
```bash
# 进入项目目录
cd /path/to/麦田智囊

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
1. 复制环境变量示例文件
```bash
cp demo/.env.example demo/.env
```

2. 编辑 `.env` 文件，填写API密钥
```
# API密钥配置
OPENAI_API_KEY=your_openai_api_key

# 其他配置
MODEL_NAME=deepseek-v2
```

### 3. 运行应用
```bash
# 进入demo目录
cd demo

# 运行Streamlit应用
streamlit run app.py
```

### 4. 访问应用
打开浏览器，访问 `http://localhost:8501`

## 使用指南

### 乡村专属极速备课
1. 在左侧选择"乡村专属极速备课"
2. 选择学科、年级
3. 输入课题
4. 填写乡村特色情境（如农田、农作物、乡村生活等）
5. 点击"开始备课"按钮
6. 等待系统生成教案（约10秒）

### 老教师经验一键传承
1. 在左侧选择"老教师经验一键传承"
2. 上传手写教案照片（支持jpg、jpeg、png格式）
3. 点击"开始识别与处理"按钮
4. 等待系统处理（约5秒/张）
5. 查看识别结果和结构化结果

## 注意事项
- 确保网络连接正常
- 确保API密钥有效
- 手写教案照片应清晰，避免光线过暗或过亮
- 首次运行时，系统会自动下载PaddleOCR模型，可能需要一些时间

## 项目结构
```
demo/
├── app.py              # 主应用文件
├── agents/             # Agent模块
│   ├── quick_lesson_prep.py  # 极速备课Agent
│   └── wisdom_transfer.py    # 智慧传承Agent
├── utils/              # 工具函数
├── assets/             # 资源文件
├── .env.example        # 环境变量示例
└── README.md           # 使用说明
```
