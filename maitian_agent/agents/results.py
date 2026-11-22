"""Typed inputs and results shared by agent execution entry points."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Intent = Literal[
    "quick_lesson_prep",
    "wisdom_transfer",
    "classroom_companion",
    "material_recommend",
    "meeting_notes",
    "general_chat",
]


class AgentError(BaseModel):
    code: str
    message: str


class AgentResult[T: BaseModel](BaseModel):
    status: Literal["success", "needs_review", "failed"]
    data: T | None
    error: AgentError | None = None

    def __contains__(self, value: str) -> bool:
        """Keep simple substring checks source-compatible for legacy callers."""
        return value in str(getattr(self.data, "text", ""))

    def __str__(self) -> str:
        return str(getattr(self.data, "text", ""))


class TextData(BaseModel):
    text: str


class WisdomData(TextData):
    review_reasons: list[Literal["OCR_REQUIRES_REVIEW", "OCR_NO_TEXT"]]


class RouteData(BaseModel):
    intent: Intent
    description: str
    available_intents: list[Intent]


class LessonPrepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    grade: str
    topic: str
    rural_context: str | None = ""


class QuizInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["quiz", "retrieve"] = "quiz"
    subject: str
    grade: str
    topic: str
    knowledge_points: str | None = ""
    question_count: int | None = 5
    question_types: str | None = "选择题、填空题"


class MaterialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    grade: str
    topic: str
    knowledge_points: str | None = ""
    rural_context: str | None = ""


class MeetingNotesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcript: str


class WisdomInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_path: str


class RouteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_input: str
