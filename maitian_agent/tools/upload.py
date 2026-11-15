"""Validate image uploads before passing a private server path to OCR."""
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from maitian_agent.tools.file import FileProcessor

IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}


async def save_image_upload(file: UploadFile, settings) -> str:
    """Read at most the configured limit plus one byte and verify decoded content."""
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in IMAGE_FORMATS:
            raise HTTPException(415, "仅支持 PNG、JPG、JPEG、WebP 图片")
        limit = settings.max_upload_mb * 1024 * 1024
        if limit <= 0:
            raise HTTPException(413, "上传大小限制无效，请联系管理员")
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, "文件过大，请上传大小限制以内的图片")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format != IMAGE_FORMATS[suffix]:
                    raise ValueError("format mismatch")
                image.verify()
            with Image.open(BytesIO(content)) as image:
                image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            raise HTTPException(415, "文件内容不是有效的对应格式图片") from None
        processor = FileProcessor(settings.upload_dir, [ext[1:] for ext in IMAGE_FORMATS])
        return processor.save_bytes(content, f"{uuid4().hex}{suffix}")
    finally:
        await file.close()
