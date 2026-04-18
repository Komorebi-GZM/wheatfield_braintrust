"""
语音识别工具
基于Whisper API
"""
from abc import ABC, abstractmethod
from typing import Optional
import os


class BaseASR(ABC):
    """语音识别基类"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """转写音频"""
        pass


class WhisperASR(BaseASR):
    """Whisper语音识别

    支持OpenAI Whisper API和本地Whisper模型
    """

    def __init__(
        self,
        model: str = "base",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        language: str = "zh"
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.language = language
        self._client = None

    @property
    def client(self):
        """获取OpenAI客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        return self._client

    def transcribe(self, audio_path: str) -> str:
        """转写音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            转写文本
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language
                )
            return transcript.text
        except Exception as e:
            raise RuntimeError(f"音频转写失败: {str(e)}")

    def transcribe_with_srt(self, audio_path: str) -> str:
        """转写音频并生成SRT字幕

        Args:
            audio_path: 音频文件路径

        Returns:
            SRT格式字幕
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    response_format="srt"
                )
            return transcript
        except Exception as e:
            raise RuntimeError(f"字幕生成失败: {str(e)}")


class LocalWhisperASR(BaseASR):
    """本地Whisper模型（待实现）"""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self):
        """加载模型"""
        # TODO: 实现本地Whisper模型加载
        raise NotImplementedError("本地Whisper模型待实现")

    def transcribe(self, audio_path: str) -> str:
        """转写音频"""
        raise NotImplementedError("本地Whisper模型待实现")
