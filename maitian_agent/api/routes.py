"""
FastAPI应用
REST API封装
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os

from agents.quick_lesson_prep import QuickLessonPrepAgent
from agents.wisdom_transfer import WisdomTransferAgent
from agents.classroom_companion import ClassroomCompanionAgent
from agents.material_agent import MaterialAgent
from agents.meeting_notes import MeetingNotesAgent
from agents.router import RouterAgent

app = FastAPI(
    title="麦田智囊API",
    description="乡村教育Agent系统REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agents = {}


@app.on_event("startup")
async def startup_event():
    """初始化Agent"""
    global agents
    agents = {
        "quick_lesson_prep": QuickLessonPrepAgent(),
        "wisdom_transfer": WisdomTransferAgent(),
        "classroom_companion": ClassroomCompanionAgent(),
        "material": MaterialAgent(),
        "meeting_notes": MeetingNotesAgent(),
        "router": RouterAgent(),
    }


class LessonPrepRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    rural_context: Optional[str] = ""


class WisdomTransferRequest(BaseModel):
    image_path: str


class QuizRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    knowledge_points: Optional[str] = ""
    question_count: Optional[int] = 5
    question_types: Optional[str] = "选择题、填空题"


class MaterialRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    knowledge_points: Optional[str] = ""
    rural_context: Optional[str] = ""


class MeetingNotesRequest(BaseModel):
    transcript: str


class RouteRequest(BaseModel):
    user_input: str


@app.get("/")
async def root():
    """根路径"""
    return {"message": "麦田智囊 API v1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "agents": list(agents.keys())}


@app.post("/api/lesson-prep")
async def lesson_prep(request: LessonPrepRequest):
    """极速备课"""
    try:
        result = agents["quick_lesson_prep"].run(
            subject=request.subject,
            grade=request.grade,
            topic=request.topic,
            rural_context=request.rural_context
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wisdom-transfer")
async def wisdom_transfer(request: WisdomTransferRequest):
    """智慧传承"""
    try:
        result = agents["wisdom_transfer"].run(image_path=request.image_path)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    """生成练习题"""
    try:
        result = agents["classroom_companion"].generate_quiz(
            subject=request.subject,
            grade=request.grade,
            topic=request.topic,
            knowledge_points=request.knowledge_points,
            question_count=request.question_count,
            question_types=request.question_types
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/material")
async def recommend_material(request: MaterialRequest):
    """素材推荐"""
    try:
        result = agents["material"].recommend_materials(
            subject=request.subject,
            grade=request.grade,
            topic=request.topic,
            knowledge_points=request.knowledge_points,
            rural_context=request.rural_context
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/meeting-notes")
async def process_meeting_notes(request: MeetingNotesRequest):
    """教研纪要"""
    try:
        result = agents["meeting_notes"].process_meeting_notes(
            meeting_transcript=request.transcript
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/route")
async def route_intent(request: RouteRequest):
    """意图路由"""
    try:
        result = agents["router"].run({"user_input": request.user_input})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    return app
