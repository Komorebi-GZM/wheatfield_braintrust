"""
Streamlit Web应用
麦田智囊乡村教育Agent系统主界面
"""
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from maitian_agent.agent_call import invoke_agent
from maitian_agent.agents.classroom_companion import ClassroomCompanionAgent
from maitian_agent.agents.material_agent import MaterialAgent
from maitian_agent.agents.meeting_notes import MeetingNotesAgent
from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.router import RouterAgent
from maitian_agent.config.settings import get_settings

load_dotenv()

st.set_page_config(
    page_title="麦田智囊 - 乡村教育Agent系统",
    page_icon="🌾",
    layout="wide"
)


def init_agents():
    """初始化Agent实例"""
    if "quick_lesson_prep_agent" not in st.session_state:
        st.session_state.quick_lesson_prep_agent = QuickLessonPrepAgent()

    if "classroom_companion_agent" not in st.session_state:
        st.session_state.classroom_companion_agent = ClassroomCompanionAgent()

    if "material_agent" not in st.session_state:
        st.session_state.material_agent = MaterialAgent()

    if "meeting_notes_agent" not in st.session_state:
        st.session_state.meeting_notes_agent = MeetingNotesAgent()

    if "router_agent" not in st.session_state:
        st.session_state.router_agent = RouterAgent()


def render_home():
    """渲染首页"""
    st.markdown("""
    ## 项目介绍
    "麦田智囊"是基于LangChain与自进化RAG的乡村教育Agent系统，致力于为乡村教师提供专属的AI教研搭档。

    ### 核心功能
    1. **乡村专属极速备课**：10秒语音生成乡土化教案
    2. **老教师经验一键传承**：手写教案OCR识别→结构化→入库→可检索
    3. **课堂实时伴教**：3秒极速出题、经典题检索
    4. **具象化素材推荐**：自动匹配科普视频/3D模型/动画素材
    5. **教研纪要生成**：录音转写→结构化报告→成果入库

    ### 技术特点
    - 基于LangChain 1.2.x（LCEL）
    - 自进化RAG知识库（Chroma + BGE-Large-zh）
    - 多模态支持（语音识别、OCR）
    - 个性化教师风格记忆
    """)


def render_quick_lesson_prep():
    """渲染极速备课页面"""
    st.markdown("## 乡村专属极速备课")
    st.markdown("通过简单的输入，10秒生成乡土化教案")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.selectbox(
            "学科",
            ["语文", "数学", "英语", "科学", "道德与法治", "音乐", "美术", "体育"]
        )
        grade = st.selectbox(
            "年级",
            ["一年级", "二年级", "三年级", "四年级", "五年级", "六年级"]
        )

    with col2:
        topic = st.text_input("课题")
        rural_context = st.text_area(
            "乡村特色情境",
            placeholder="例如：结合当地种植的水稻、小麦等农作物，或乡村生活实例"
        )

    if st.button("开始备课", type="primary"):
        if not topic:
            st.error("请输入课题")
        else:
            with st.spinner("正在生成教案，请稍候..."):
                agent = st.session_state.quick_lesson_prep_agent
                result = invoke_agent(
                    agent.run,
                    error_handler=st.error,
                    subject=subject,
                    grade=grade,
                    topic=topic,
                    rural_context=rural_context,
                )
                if result is not None:
                    st.success("教案生成完成！")
                    st.markdown("### 生成的教案")
                    st.markdown(result)

    st.markdown("---")
    st.markdown("#### 备课助手功能说明")
    st.info("""
    - 输入学科、年级、课题和乡村特色情境
    - 系统将生成结合乡土元素的详细教案
    - 教案包含：教学目标、重难点、教学准备、教学过程、作业布置、板书设计
    """)


def render_wisdom_transfer():
    """渲染智慧传承页面"""
    st.markdown("## 老教师经验一键传承")
    st.markdown("上传手写教案照片，系统自动识别、结构化并入库")

    uploaded_file = st.file_uploader(
        "上传手写教案照片",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="上传的手写教案", width=300)

        with col2:
            if st.button("开始识别与处理", type="primary"):
                with st.spinner("正在处理，请稍候..."):
                    try:
                        response = httpx.post(
                            os.getenv("MAITIAN_API_BASE", "http://127.0.0.1:8000").rstrip("/")
                            + "/api/wisdom-transfer",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(),
                                            uploaded_file.type)},
                            timeout=120,
                        )
                        payload = response.json()
                        if response.is_error:
                            detail = payload.get("detail")
                            st.error(detail if isinstance(detail, str) else "上传失败，请检查图片")
                        else:
                            st.success("处理完成！")
                            st.markdown("### 识别结果")
                            st.markdown(payload["result"])
                    except (httpx.HTTPError, ValueError, KeyError):
                        st.error("服务暂不可用，请稍后重试")

    st.markdown("---")
    st.markdown("#### 功能说明")
    st.info("""
    - 支持上传JPG、PNG格式的手写教案照片
    - 使用EasyOCR自动识别手写文字
    - 大模型将识别结果结构化处理
    - 可将结果保存到知识库中
    """)


def render_classroom_companion():
    """渲染课堂伴教页面"""
    st.markdown("## 课堂实时伴教")
    st.markdown("3秒极速出题、经典题检索")

    action = st.radio(
        "选择功能",
        ["生成练习题", "检索经典题"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    with col1:
        subject = st.selectbox("学科", ["语文", "数学", "英语", "科学"])
        grade = st.selectbox("年级", ["一年级", "二年级", "三年级", "四年级", "五年级", "六年级"])

    with col2:
        topic = st.text_input("课题")
        knowledge_points = st.text_area("知识点", placeholder="输入要考察的知识点")

    if action == "生成练习题":
        question_count = st.slider("题目数量", 1, 10, 5)
        question_types = st.multiselect(
            "题目类型",
            ["选择题", "填空题", "判断题", "简答题"],
            default=["选择题", "填空题"]
        )

        if st.button("生成练习题", type="primary"):
            if not topic:
                st.error("请输入课题")
            else:
                with st.spinner("正在生成练习题..."):
                    agent = st.session_state.classroom_companion_agent
                    result = invoke_agent(
                        agent.generate_quiz,
                        error_handler=st.error,
                        subject=subject,
                        grade=grade,
                        topic=topic,
                        knowledge_points=knowledge_points,
                        question_count=question_count,
                        question_types="、".join(question_types),
                    )
                    if result is not None:
                        st.success("练习题生成完成！")
                        st.markdown("### 生成的练习题")
                        st.markdown(result)

    else:
        if st.button("检索经典题", type="primary"):
            if not topic:
                st.error("请输入课题")
            else:
                with st.spinner("正在检索经典题..."):
                    agent = st.session_state.classroom_companion_agent
                    result = invoke_agent(
                        agent.retrieve_classic_questions,
                        error_handler=st.error,
                        subject=subject,
                        grade=grade,
                        topic=topic,
                        knowledge_points=knowledge_points,
                    )
                    if result is not None:
                        st.success("检索完成！")
                        st.markdown("### 经典题检索结果")
                        st.markdown(result)


def render_material_recommend():
    """渲染素材推荐页面"""
    st.markdown("## 具象化素材推荐")
    st.markdown("自动匹配科普视频/3D模型/动画素材")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.selectbox("学科", ["语文", "数学", "英语", "科学", "道德与法治"])
        grade = st.selectbox("年级", ["一年级", "二年级", "三年级", "四年级", "五年级", "六年级"])

    with col2:
        topic = st.text_input("课题")
        knowledge_points = st.text_area("知识点", placeholder="输入要讲解的知识点")

    rural_context = st.text_area("乡村情境", placeholder="结合乡村实际情境")

    if st.button("推荐素材", type="primary"):
        if not topic:
            st.error("请输入课题")
        else:
            with st.spinner("正在推荐素材..."):
                agent = st.session_state.material_agent
                result = invoke_agent(
                    agent.recommend_materials,
                    error_handler=st.error,
                    subject=subject,
                    grade=grade,
                    topic=topic,
                    knowledge_points=knowledge_points,
                    rural_context=rural_context,
                )
                if result is not None:
                    st.success("素材推荐完成！")
                    st.markdown("### 推荐素材")
                    st.markdown(result)

    st.markdown("---")
    st.markdown("#### 功能说明")
    st.info("""
    - 根据教学内容推荐适合的具象化素材
    - 素材类型包括：科普视频、3D模型、动画演示、图片素材
    - 素材贴近乡村学生的生活经验
    """)


def render_meeting_notes():
    """渲染教研纪要页面"""
    st.markdown("## 教研纪要生成")
    st.markdown("录音转写→结构化教研报告→成果入库")

    st.markdown("#### 输入方式")

    input_method = st.radio(
        "选择输入方式",
        ["直接输入会议记录", "上传音频文件（待开发）"],
        horizontal=True
    )

    if input_method == "直接输入会议记录":
        meeting_transcript = st.text_area(
            "粘贴会议记录",
            height=300,
            placeholder="粘贴教研会议的原始记录文本..."
        )

        if st.button("生成结构化报告", type="primary"):
            if not meeting_transcript:
                st.error("请输入会议记录")
            else:
                with st.spinner("正在处理，请稍候..."):
                    agent = st.session_state.meeting_notes_agent
                    result = invoke_agent(
                        agent.process_meeting_notes,
                        meeting_transcript,
                        error_handler=st.error,
                    )
                    if result is not None:
                        st.success("报告生成完成！")
                        st.markdown("### 教研纪要")
                        st.markdown(result)
    else:
        st.info("音频转写功能待集成Whisper API")

    st.markdown("---")
    st.markdown("#### 功能说明")
    st.info("""
    - 支持直接粘贴会议记录或上传音频
    - 自动生成结构化的教研报告
    - 包含会议议题、讨论摘要、决议事项、行动计划
    - 可将教研成果保存到知识库
    """)


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 关于")
    st.sidebar.markdown("""
    **麦田智囊 v1.0.0**

    基于LangChain与自进化RAG的乡村教育Agent系统

    © 2026 麦田智囊团队
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 环境信息")
    st.sidebar.markdown(f"- 项目版本：{os.getenv('VERSION', '1.0.0')}")
    llm_model = get_settings().llm_model or "未配置"
    st.sidebar.markdown(f"- 模型：{llm_model}")


def main():
    """主函数"""
    init_agents()

    st.title("🌾 麦田智囊 - 乡村教育Agent系统")
    st.markdown("### 乡村教师专属AI教研搭档")

    page_names_to_funcs = {
        "🏠 首页": render_home,
        "📚 极速备课": render_quick_lesson_prep,
        "📝 智慧传承": render_wisdom_transfer,
        "🎯 课堂伴教": render_classroom_companion,
        "🎬 素材推荐": render_material_recommend,
        "📋 教研纪要": render_meeting_notes,
    }

    selected_page = st.sidebar.selectbox(
        "选择功能",
        list(page_names_to_funcs.keys())
    )

    page_names_to_funcs[selected_page]()
    render_sidebar()


if __name__ == "__main__":
    main()
