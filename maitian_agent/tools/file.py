"""
文件处理工具
"""
from typing import List, Optional, Dict, Any
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime


class FileProcessor:
    """文件处理器"""

    def __init__(
        self,
        upload_dir: str = "./data/uploads",
        allowed_extensions: List[str] = None
    ):
        self.upload_dir = upload_dir
        self.allowed_extensions = allowed_extensions or [
            "jpg", "jpeg", "png", "mp3", "mp4", "wav", "m4a", "pdf", "docx"
        ]
        os.makedirs(upload_dir, exist_ok=True)

    def save_upload(self, file, custom_filename: Optional[str] = None) -> str:
        """保存上传的文件

        Args:
            file: 上传的文件对象
            custom_filename: 自定义文件名

        Returns:
            保存的文件路径
        """
        if custom_filename:
            filename = custom_filename
        else:
            ext = file.name.split(".")[-1] if "." in file.name else ""
            filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"

        file_path = os.path.join(self.upload_dir, filename)

        if hasattr(file, "getbuffer"):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        else:
            shutil.copy(file, file_path)

        return file_path

    def save_bytes(self, content: bytes, filename: str) -> str:
        """保存字节内容

        Args:
            content: 字节内容
            filename: 文件名

        Returns:
            保存的文件路径
        """
        file_path = os.path.join(self.upload_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            文件信息字典
        """
        if not os.path.exists(file_path):
            return {}

        stat = os.stat(file_path)
        return {
            "filename": os.path.basename(file_path),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": Path(file_path).suffix
        }

    def delete_file(self, file_path: str) -> bool:
        """删除文件

        Args:
            file_path: 文件路径

        Returns:
            是否删除成功
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def list_files(
        self,
        extension: Optional[str] = None,
        sort_by: str = "modified"
    ) -> List[Dict[str, Any]]:
        """列出文件

        Args:
            extension: 文件扩展名过滤
            sort_by: 排序方式 (modified/created/size/name)

        Returns:
            文件列表
        """
        files = []
        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.isfile(file_path):
                if extension and not filename.endswith(f".{extension}"):
                    continue
                files.append(self.get_file_info(file_path))

        if sort_by == "modified":
            files.sort(key=lambda x: x["modified_at"], reverse=True)
        elif sort_by == "created":
            files.sort(key=lambda x: x["created_at"], reverse=True)
        elif sort_by == "size":
            files.sort(key=lambda x: x["size"], reverse=True)
        elif sort_by == "name":
            files.sort(key=lambda x: x["filename"])

        return files

    def cleanup_old_files(self, days: int = 7) -> int:
        """清理旧文件

        Args:
            days: 保留天数

        Returns:
            删除的文件数量
        """
        count = 0
        now = datetime.now().timestamp()
        threshold = days * 24 * 60 * 60

        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.isfile(file_path):
                if now - os.stat(file_path).st_mtime > threshold:
                    os.remove(file_path)
                    count += 1

        return count
