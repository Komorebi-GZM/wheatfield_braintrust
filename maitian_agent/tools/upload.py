"""Validate image uploads before passing a private server path to OCR."""

import logging
import time
import warnings
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from maitian_agent.tools.file import FileProcessor

logger = logging.getLogger(__name__)
MAX_UPLOAD_IMAGE_PIXELS = 8_000_000
UPLOAD_TTL_SECONDS = 24 * 60 * 60

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
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    if image.width * image.height > MAX_UPLOAD_IMAGE_PIXELS:
                        raise HTTPException(413, "图片像素过多，请缩小图片后重试")
                with Image.open(BytesIO(content)) as image:
                    if image.format != IMAGE_FORMATS[suffix]:
                        raise ValueError("format mismatch")
                    image.verify()
                with Image.open(BytesIO(content)) as image:
                    image.load()
        except (Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise HTTPException(413, "图片像素过多，请缩小图片后重试") from None
        except (OSError, ValueError, SyntaxError):
            raise HTTPException(415, "文件内容不是有效的对应格式图片") from None
        processor = FileProcessor(settings.upload_dir, [ext[1:] for ext in IMAGE_FORMATS])
        return processor.save_bytes(content, f"{uuid4().hex}{suffix}")
    finally:
        await file.close()


def delete_saved_upload(path: str, upload_dir: str) -> None:
    """Delete a saved upload only within the resolved upload directory."""
    base = Path(upload_dir).resolve()
    target = Path(path).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError("上传文件路径超出上传目录")
    target.unlink()


def cleanup_stale_uploads(upload_dir: str, max_age_seconds: float = UPLOAD_TTL_SECONDS) -> int:
    """Remove expired regular files directly inside the upload directory."""
    base = Path(upload_dir).resolve()
    if not base.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    count = 0
    for path in base.iterdir():
        try:
            # Never follow symlinks or recurse into directories during cleanup.
            if not path.is_symlink() and path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                count += 1
        except OSError:
            logger.exception("清理过期上传文件失败")
    return count
