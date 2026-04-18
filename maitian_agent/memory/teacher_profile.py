"""
教师画像管理
长期记忆实现
"""
from typing import Dict, Any, List, Optional
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os


class TeachingStyle(BaseModel):
    """教学风格标签"""
    style_name: str = Field(description="风格名称")
    description: str = Field(description="风格描述")
    examples: List[str] = Field(default_factory=list, description="示例")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style_name": self.style_name,
            "description": self.description,
            "examples": self.examples
        }


class TeacherProfile(BaseModel):
    """教师画像

    长期记忆：教师专属画像，风格标签向量化存储
    """

    teacher_id: str = Field(description="教师ID")
    name: str = Field(description="教师姓名")
    school: str = Field(default="", description="所在学校")
    subjects: List[str] = Field(default_factory=list, description="教授科目")
    grades: List[str] = Field(default_factory=list, description="教授年级")
    teaching_years: int = Field(default=0, description="教龄")

    teaching_styles: List[TeachingStyle] = Field(default_factory=list, description="教学风格")
    rural_experience: str = Field(default="", description="乡村教学经验")
    specializations: List[str] = Field(default_factory=list, description="专业特长")

    preferences: Dict[str, Any] = Field(default_factory=dict, description="个性化偏好")
    lesson_plans_count: int = Field(default=0, description="教案数量")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "teacher_id": self.teacher_id,
            "name": self.name,
            "school": self.school,
            "subjects": self.subjects,
            "grades": self.grades,
            "teaching_years": self.teaching_years,
            "teaching_styles": [style.to_dict() for style in self.teaching_styles],
            "rural_experience": self.rural_experience,
            "specializations": self.specializations,
            "preferences": self.preferences,
            "lesson_plans_count": self.lesson_plans_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    def update(self, **kwargs) -> None:
        """更新画像"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()

    def add_teaching_style(self, style: TeachingStyle) -> None:
        """添加教学风格"""
        self.teaching_styles.append(style)
        self.updated_at = datetime.now().isoformat()

    def increment_lesson_plans(self) -> None:
        """增加教案计数"""
        self.lesson_plans_count += 1
        self.updated_at = datetime.now().isoformat()


class TeacherProfileManager:
    """教师画像管理器"""

    def __init__(self, persist_directory: str = "./data/profiles"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

    def save_profile(self, profile: TeacherProfile) -> None:
        """保存教师画像

        Args:
            profile: 教师画像
        """
        profile_file = os.path.join(
            self.persist_directory,
            f"profile_{profile.teacher_id}.json"
        )

        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

    def load_profile(self, teacher_id: str) -> Optional[TeacherProfile]:
        """加载教师画像

        Args:
            teacher_id: 教师ID

        Returns:
            教师画像，如果没有找到则返回None
        """
        profile_file = os.path.join(
            self.persist_directory,
            f"profile_{teacher_id}.json"
        )

        if not os.path.exists(profile_file):
            return None

        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            styles = [
                TeachingStyle(**style_data)
                for style_data in data.get("teaching_styles", [])
            ]
            data["teaching_styles"] = styles

            return TeacherProfile(**data)
        except Exception:
            return None

    def create_profile(
        self,
        teacher_id: str,
        name: str,
        school: str = "",
        subjects: List[str] = None,
        grades: List[str] = None,
        **kwargs
    ) -> TeacherProfile:
        """创建新教师画像

        Args:
            teacher_id: 教师ID
            name: 教师姓名
            school: 所在学校
            subjects: 教授科目
            grades: 教授年级
            **kwargs: 其他参数

        Returns:
            新建的教师画像
        """
        profile = TeacherProfile(
            teacher_id=teacher_id,
            name=name,
            school=school,
            subjects=subjects or [],
            grades=grades or [],
            **kwargs
        )
        self.save_profile(profile)
        return profile

    def list_profiles(self) -> List[Dict[str, Any]]:
        """列出所有教师画像摘要

        Returns:
            画像摘要列表
        """
        profiles = []
        for filename in os.listdir(self.persist_directory):
            if filename.startswith("profile_") and filename.endswith(".json"):
                profile_path = os.path.join(self.persist_directory, filename)
                try:
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        profiles.append({
                            "teacher_id": data.get("teacher_id"),
                            "name": data.get("name"),
                            "school": data.get("school"),
                            "subjects": data.get("subjects", []),
                            "updated_at": data.get("updated_at")
                        })
                except Exception:
                    continue
        return profiles

    def delete_profile(self, teacher_id: str) -> bool:
        """删除教师画像

        Args:
            teacher_id: 教师ID

        Returns:
            是否删除成功
        """
        profile_file = os.path.join(
            self.persist_directory,
            f"profile_{teacher_id}.json"
        )

        if os.path.exists(profile_file):
            os.remove(profile_file)
            return True
        return False


class TeacherStyleVectorStore:
    """教师风格向量存储（待实现）"""

    def __init__(
        self,
        embedding_model: Embeddings,
        persist_directory: str = "./data/style_vectors"
    ):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        # TODO: 集成向量存储
        raise NotImplementedError("教师风格向量存储待实现")

    def add_teacher_style(self, teacher_id: str, style_description: str) -> None:
        """添加教师风格向量"""
        raise NotImplementedError("教师风格向量存储待实现")

    def find_similar_teachers(self, style_description: str, k: int = 5) -> List[str]:
        """查找风格相似的教师"""
        raise NotImplementedError("教师风格向量存储待实现")
