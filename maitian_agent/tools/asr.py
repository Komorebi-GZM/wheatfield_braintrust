"""
语音识别工具
基于 OpenAI 兼容的 Whisper API（厂商无关）
"""

from abc import ABC, abstractmethod
from typing import cast

from maitian_agent.config.settings import get_settings


class ASRNotConfiguredError(RuntimeError):
    """ASR 凭证未配置时抛出；构造前就报错，绝不把 None 传给 SDK。"""

    code = "ASR_NOT_CONFIGURED"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or ("未配置语音识别，请设置 ASR_BASE_URL、ASR_MODEL、ASR_API_KEY")
        super().__init__(self.message)


class BaseASR(ABC):
    """语音识别基类"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """转写音频"""


class WhisperASR(BaseASR):
    """Whisper 语音识别（OpenAI 兼容远程 API）

    三项配置 ``ASR_BASE_URL`` / ``ASR_MODEL`` / ``ASR_API_KEY`` 必填，均无厂商默认值。
    未显式传参时从 settings 读取；任一缺失即报错，不创建 OpenAI 客户端。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str = "zh",
    ):
        settings = get_settings()
        resolved_model = model if model is not None else settings.asr_model
        resolved_api_key = api_key if api_key is not None else settings.get_asr_api_key()
        resolved_api_base = api_base if api_base is not None else settings.asr_base_url

        missing = [
            name
            for name, value in (
                ("ASR_BASE_URL", resolved_api_base),
                ("ASR_MODEL", resolved_model),
                ("ASR_API_KEY", resolved_api_key),
            )
            if not value
        ]
        if missing:
            raise ASRNotConfiguredError(
                "语音识别配置不完整，缺少："
                + "、".join(missing)
                + "。请设置 ASR_BASE_URL、ASR_MODEL、ASR_API_KEY。"
            )

        self.model: str = cast(str, resolved_model)
        self.api_key: str = cast(str, resolved_api_key)
        self.api_base: str = cast(str, resolved_api_base)
        self.language = language
        self._client = None

    @property
    def client(self):
        """获取 OpenAI 客户端（显式传入 api_key 与 base_url，不读环境变量）"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
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
                    model=self.model, file=audio_file, language=self.language
                )
            return transcript.text
        except Exception as e:
            raise RuntimeError(f"音频转写失败: {e!s}")

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
                    model=self.model, file=audio_file, language=self.language, response_format="srt"
                )
            return transcript
        except Exception as e:
            raise RuntimeError(f"字幕生成失败: {e!s}")


class LocalWhisperASR(BaseASR):
    """本地Whisper模型（待实现）"""

    def __init__(self, model_name: str = "base", device: str = "cpu"):
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
