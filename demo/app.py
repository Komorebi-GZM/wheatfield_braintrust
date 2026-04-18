import streamlit as st
import os
from dotenv import load_dotenv
from agents.quick_lesson_prep import QuickLessonPrepAgent
from agents.wisdom_transfer import WisdomTransferAgent

# 加载环境变量
load_dotenv()

# 确保assets目录存在
if not os.path.exists("assets"):
    os.makedirs("assets")

# 设置页面配置
st.set_page_config(
    page_title="麦田智囊 - 乡村教育Agent系统",
    page_icon="🌾",
    layout="wide"
)

# 页面标题
st.title("🌾 麦田智囊 - 乡村教育Agent系统")
st.markdown("### 乡村教师专属AI教研搭档")

# 功能选择
option = st.sidebar.selectbox(
    "选择功能",
    ["首页", "乡村专属极速备课", "老教师经验一键传承"]
)

if option == "首页":
    st.markdown("""
    ## 项目介绍
    "麦田智囊"是基于LangChain与自进化RAG的乡村教育Agent系统，致力于为乡村教师提供专属的AI教研搭档。
    
    ### 核心功能
    1. **乡村专属极速备课**：10秒语音生成乡土化教案
    2. **老教师经验一键传承**：手写教案OCR识别→结构化→入库→可检索
    
    ### 技术特点
    - 基于LangChain 0.2.x（LCEL）
    - 自进化RAG知识库（Chroma + BGE-Large-zh）
    - 多模态支持（语音识别、OCR）
    - 个性化教师风格记忆
    """)

elif option == "乡村专属极速备课":
    st.markdown("## 乡村专属极速备课")
    st.markdown("通过语音指令，10秒生成乡土化教案")
    
    # 输入区域
    subject = st.selectbox("学科", ["语文", "数学", "英语", "科学", "道德与法治"])
    grade = st.selectbox("年级", ["一年级", "二年级", "三年级", "四年级", "五年级", "六年级"])
    topic = st.text_input("课题")
    rural_context = st.text_area("乡村特色情境（如农田、农作物、乡村生活等）", placeholder="例如：结合当地种植的水稻、小麦等农作物")
    
    if st.button("开始备课"):
        if not topic:
            st.error("请输入课题")
        else:
            with st.spinner("正在生成教案..."):
                agent = QuickLessonPrepAgent()
                result = agent.run(subject, grade, topic, rural_context)
                st.success("教案生成完成！")
                st.markdown("### 生成的教案")
                st.markdown(result)

elif option == "老教师经验一键传承":
    st.markdown("## 老教师经验一键传承")
    st.markdown("上传手写教案照片，系统自动识别、结构化并入库")
    
    # 上传区域
    uploaded_file = st.file_uploader("上传手写教案照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 保存上传的文件
        file_path = os.path.join("assets", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("开始识别与处理"):
            with st.spinner("正在处理..."):
                agent = WisdomTransferAgent()
                result = agent.run(file_path)
                st.success("处理完成！")
                st.markdown("### 识别结果")
                st.markdown(result)

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 麦田智囊")
