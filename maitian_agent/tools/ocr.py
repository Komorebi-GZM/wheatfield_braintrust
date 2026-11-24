"""
OCR工具
支持EasyOCR和PaddleOCR
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import os
import sys
from pathlib import Path


def _ensure_ocr_module_path() -> None:
    """把仓库根目录加入 sys.path，使 `ocr_module.ocr.engine` 可被导入。"""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class BaseOCR(ABC):
    """OCR基类"""

    @abstractmethod
    def recognize(self, image_path: str) -> str:
        """识别文字"""
        pass

    @abstractmethod
    def recognize_batch(self, image_paths: List[str]) -> List[str]:
        """批量识别"""
        pass


class EasyOCRProcessor(BaseOCR):
    """EasyOCR处理器"""

    def __init__(
        self,
        languages: List[str] = ["ch_sim", "en"],
        gpu: bool = False,
        model_storage_directory: Optional[str] = None
    ):
        self.languages = languages
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self._reader = None

    @property
    def reader(self):
        """获取OCR读取器"""
        if self._reader is None:
            from easyocr import Reader
            self._reader = Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory
            )
        return self._reader

    def recognize(self, image_path: str, detail: int = 1) -> str:
        """识别文字

        Args:
            image_path: 图片路径
            detail: 返回详情级别 0=仅文字, 1=带位置信息

        Returns:
            识别的文字
        """
        result = self.reader.readtext(image_path, detail=detail)

        if detail == 0:
            return "".join([item[1] for item in result])
        else:
            text_lines = []
            for detection in result:
                text_lines.append(detection[1])
            return "\n".join(text_lines)

    def recognize_with_boxes(self, image_path: str) -> List[Tuple[List, str, float]]:
        """识别文字并返回位置信息

        Args:
            image_path: 图片路径

        Returns:
            [(边界框, 文字, 置信度), ...]
        """
        result = self.reader.readtext(image_path, detail=1)
        return [(item[0], item[1], item[2]) for item in result]

    def recognize_batch(self, image_paths: List[str]) -> List[str]:
        """批量识别

        Args:
            image_paths: 图片路径列表

        Returns:
            识别结果列表
        """
        return [self.recognize(path, detail=0) for path in image_paths]


class PaddleOCRProcessor(BaseOCR):
    """PaddleOCR处理器（待实现）"""

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = 'ch',
        enable_mkldnn: bool = False,
        cpu_threads: int = 10
    ):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads
        self._ocr = None

    @property
    def ocr(self):
        """获取OCR实例"""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                enable_mkldnn=self.enable_mkldnn,
                cpu_threads=self.cpu_threads
            )
        return self._ocr

    def recognize(self, image_path: str) -> str:
        """识别文字"""
        result = self.ocr.ocr(image_path, cls=True)

        text_lines = []
        for line in result[0]:
            text_lines.append(line[1][0])
        return "\n".join(text_lines)

    def recognize_batch(self, image_paths: List[str]) -> List[str]:
        """批量识别"""
        return [self.recognize(path) for path in image_paths]


class XiaomiStructuredOCR(BaseOCR):
    """基于 ocr_module（Xiaomi-OCR-0 QLoRA）的结构化 OCR，直接输出 Markdown。

    输出已含结构（标题/列表/表格/LaTeX 公式），故下游无需再做 LLM 结构化。
    """
    is_structured = True

    def __init__(
        self,
        base_model: Optional[str] = None,
        adapter_dir: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 4096,
    ):
        self.base_model = base_model or "SeerRay-Lab/Xiaomi-OCR-0"
        self.adapter_dir = adapter_dir
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            _ensure_ocr_module_path()
            from ocr_module.ocr.engine import OCREngine  # 延迟导入，避免无 torch 时报错
            self._engine = OCREngine(
                base_model=self.base_model,
                adapter_dir=self.adapter_dir,
                load_in_4bit=self.load_in_4bit,
                max_new_tokens=self.max_new_tokens,
            )
        return self._engine

    def recognize(self, image_path: str, **kwargs) -> str:
        return self.engine.recognize_page(image_path)["markdown"]

    def recognize_with_boxes(self, image_path: str):
        raise NotImplementedError("XiaomiStructuredOCR 不返回字符级边界框")

    def recognize_batch(self, image_paths: List[str]) -> List[str]:
        return [self.recognize(p) for p in image_paths]


class OCRProcessor:
    """OCR处理器工厂"""

    @staticmethod
    def create(
        ocr_type: str = "easyocr",
        **kwargs
    ) -> BaseOCR:
        """创建OCR处理器

        Args:
            ocr_type: OCR类型 (easyocr/paddleocr)
            **kwargs: 其他参数

        Returns:
            OCR处理器实例
        """
        if ocr_type == "easyocr":
            return EasyOCRProcessor(
                languages=kwargs.get("languages", ["ch_sim", "en"]),
                gpu=kwargs.get("gpu", False),
                model_storage_directory=kwargs.get("model_storage_directory")
            )
        elif ocr_type == "paddleocr":
            return PaddleOCRProcessor(
                use_angle_cls=kwargs.get("use_angle_cls", True),
                lang=kwargs.get("lang", "ch"),
                enable_mkldnn=kwargs.get("enable_mkldnn", False),
                cpu_threads=kwargs.get("cpu_threads", 10)
            )
        elif ocr_type in ("xiaomi", "xiaomi_structured"):
            return XiaomiStructuredOCR(
                base_model=kwargs.get("base_model"),
                adapter_dir=kwargs.get("adapter_dir"),
                load_in_4bit=kwargs.get("load_in_4bit", True),
                max_new_tokens=kwargs.get("max_new_tokens", 4096),
            )
        else:
            raise ValueError(f"不支持的OCR类型: {ocr_type}")
