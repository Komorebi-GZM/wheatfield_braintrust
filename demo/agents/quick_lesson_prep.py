from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class QuickLessonPrepAgent:
    def __init__(self):
        # 从环境变量获取API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME", "deepseek-v2")
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key
        )
        
        # 备课模板
        self.lesson_plan_template = ChatPromptTemplate.from_template(
            """
            你是一位经验丰富的乡村教师，擅长结合乡村实际情境设计教案。请根据以下信息生成一份详细的教案：
            
            学科：{subject}
            年级：{grade}
            课题：{topic}
            乡村特色情境：{rural_context}
            
            教案应包含以下部分：
            1. 教学目标
            2. 教学重难点
            3. 教学准备
            4. 教学过程（导入、新授、练习、总结）
            5. 作业布置
            6. 板书设计
            
            要求：
            - 结合乡村实际情境，使用贴近农村生活的例子
            - 语言通俗易懂，符合该年级学生的认知水平
            - 教案结构清晰，可直接用于课堂教学
            - 融入乡土元素，增强学生的学习兴趣
            """
        )
    
    def run(self, subject, grade, topic, rural_context):
        """执行备课流程"""
        try:
            # 构建消息
            messages = self.lesson_plan_template.format_messages(
                subject=subject,
                grade=grade,
                topic=topic,
                rural_context=rural_context
            )
            
            # 调用大模型
            result = self.llm.invoke(messages)
            return result.content
        except Exception as e:
            return f"生成教案时出错：{str(e)}"
