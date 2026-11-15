"""
Agent模块
包含五大核心Agent链
"""

from maitian_agent.agents.base import BaseAgent
from maitian_agent.agents.classroom_companion import ClassroomCompanionAgent
from maitian_agent.agents.material_agent import MaterialAgent
from maitian_agent.agents.meeting_notes import MeetingNotesAgent
from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.router import RouterAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent

__all__ = [
    "BaseAgent",
    "QuickLessonPrepAgent",
    "WisdomTransferAgent",
    "ClassroomCompanionAgent",
    "MaterialAgent",
    "MeetingNotesAgent",
    "RouterAgent",
]
