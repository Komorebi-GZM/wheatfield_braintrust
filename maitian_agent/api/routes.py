"""
FastAPI应用
REST API封装
"""
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from maitian_agent.agents.classroom_companion import ClassroomCompanionAgent
from maitian_agent.agents.material_agent import MaterialAgent
from maitian_agent.agents.meeting_notes import MeetingNotesAgent
from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.router import RouterAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.config.settings import get_settings
from maitian_agent.tools.upload import save_image_upload

router = APIRouter()
logger = logging.getLogger(__name__)

agents = {}


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
    rural_context: str | None = ""


class QuizRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    knowledge_points: str | None = ""
    question_count: int | None = 5
    question_types: str | None = "选择题、填空题"


class MaterialRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    knowledge_points: str | None = ""
    rural_context: str | None = ""


class MeetingNotesRequest(BaseModel):
    transcript: str


class RouteRequest(BaseModel):
    user_input: str


@router.get("/")
async def root():
    """根路径"""
    return {"message": "麦田智囊 API v1.0.0", "status": "running"}


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "agents": list(agents.keys())}


@router.post("/api/lesson-prep")
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@router.post("/api/wisdom-transfer")
async def wisdom_transfer(request: Request, file: UploadFile):
    """智慧传承"""
    try:
        image_path = await save_image_upload(file, request.app.state.settings)
        result = agents["wisdom_transfer"].run(image_path=image_path)
        # The legacy Agent wraps failures in strings; never expose those details.
        if isinstance(result, str) and result.startswith("处理手写教案时出错"):
            logger.error("智慧传承处理失败: %s", result)
            raise HTTPException(500, "服务内部错误，请稍后重试")
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@router.post("/api/quiz")
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@router.post("/api/material")
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@router.post("/api/meeting-notes")
async def process_meeting_notes(request: MeetingNotesRequest):
    """教研纪要"""
    try:
        result = agents["meeting_notes"].process_meeting_notes(
            meeting_transcript=request.transcript
        )
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@router.post("/api/route")
async def route_intent(request: RouteRequest):
    """意图路由"""
    try:
        result = agents["router"].run({"user_input": request.user_input})
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("API 请求处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试") from None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield


def create_app() -> FastAPI:
    """使用当前缓存配置创建独立应用。"""
    settings = get_settings()
    app = FastAPI(title="麦田智囊API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",")
                       if origin.strip() and origin.strip() != "*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
