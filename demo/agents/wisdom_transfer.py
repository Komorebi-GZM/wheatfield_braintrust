from easyocr import Reader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class WisdomTransferAgent:
    def __init__(self):
        # 初始化OCR模型（使用EasyOCR）
        self.ocr = Reader(['ch_sim', 'en'], gpu=False)
        
        # 从环境变量获取API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME", "deepseek-v2")
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key
        )
        
        # 结构化模板
        self.structure_template = ChatPromptTemplate.from_template(
            """
            你是一位教育专家，请将以下手写教案的OCR识别结果结构化处理：
            
            {ocr_text}
            
            请按照以下格式进行结构化：
            1. 学科：
            2. 年级：
            3. 课题：
            4. 教学目标：
            5. 教学重难点：
            6. 教学过程：
            7. 作业布置：
            8. 教学反思：
            
            要求：
            - 提取关键信息，忽略无关内容
            - 保持原文意，语言通顺
            - 结构清晰，层次分明
            """
        )
    
    def run(self, image_path):
        """执行手写教案识别与结构化"""
        try:
            # 执行OCR识别
            result = self.ocr.readtext(image_path, detail=1)
            
            # 提取识别文本
            ocr_text = ""
            for detection in result:
                ocr_text += detection[1] + "\n"
            
            # 构建消息
            messages = self.structure_template.format_messages(ocr_text=ocr_text)
            
            # 结构化处理
            structured_result = self.llm.invoke(messages)
            
            return f"### OCR识别结果\n{ocr_text}\n\n### 结构化结果\n{structured_result.content}"
        except Exception as e:
            return f"处理手写教案时出错：{str(e)}"
