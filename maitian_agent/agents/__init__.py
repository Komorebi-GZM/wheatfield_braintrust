"""
Agent模块
包含五大核心Agent链
"""
from agents.base import BaseAgent
from agents.quick_lesson_prep import QuickLessonPrepAgent
from agents.wisdom_transfer import WisdomTransferAgent
from agents.classroom_companion import ClassroomCompanionAgent
from agents.material_agent import MaterialAgent
from agents.meeting_notes import MeetingNotesAgent
from agents.router import RouterAgent

__all__ = [
    "BaseAgent",
    "QuickLessonPrepAgent",
    "WisdomTransferAgent",
    "ClassroomCompanionAgent",
    "MaterialAgent",
    "MeetingNotesAgent",
    "RouterAgent",
]
