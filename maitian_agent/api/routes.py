"""FastAPI endpoints with one execution path and versioned serializers."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import get_route_path

from maitian_agent.agents.classroom_companion import ClassroomCompanionAgent
from maitian_agent.agents.material_agent import MaterialAgent
from maitian_agent.agents.meeting_notes import MeetingNotesAgent
from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.results import (
    AgentResult,
    Intent,
    LessonPrepInput,
    MaterialInput,
    MeetingNotesInput,
    QuizInput,
    RouteData,
    RouteInput,
    TextData,
    WisdomData,
    WisdomInput,
)
from maitian_agent.agents.router import RouterAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.api.upload_limit import UploadLimitMiddleware
from maitian_agent.config.settings import get_settings
from maitian_agent.llm import LLMNotConfiguredError
from maitian_agent.openapi import application_version, openapi_document
from maitian_agent.tools.upload import cleanup_stale_uploads, delete_saved_upload, save_image_upload

router = APIRouter()
logger = logging.getLogger(__name__)
agents = {}


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


class HealthResponse(BaseModel):
    status: str
    agents: list[str]
    llm_configured: bool


class RootResponse(BaseModel):
    message: str
    status: str


class LegacyTextResponse(BaseModel):
    success: Literal[True]
    result: str


class LegacyWisdomResponse(BaseModel):
    success: Literal[True]
    result: str
    status: Literal["needs_review"]


class LegacyRouteResponse(BaseModel):
    success: Literal[True]
    intent: Intent
    agent: str
    info: RouteData


class ErrorData(BaseModel):
    code: str
    message: str


class Meta(BaseModel):
    model: str | None
    duration_ms: int = Field(ge=0)
    version: str = Field(min_length=1)


class SuccessEnvelope(BaseModel):
    """status success/needs_review requires data and forbids error."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": "For success or needs_review, data is required and error is null."
        }
    )
    status: Literal["success", "needs_review"]
    data: TextData | WisdomData | RouteData
    error: None
    meta: Meta


class FailedEnvelope(BaseModel):
    """Failed responses require error and forbid data."""

    model_config = ConfigDict(
        json_schema_extra={"description": "For failed, data is null and error is required."}
    )
    status: Literal["failed"]
    data: None
    error: ErrorData
    meta: Meta


def _success_envelope_for(data_type):
    return create_model(
        f"{data_type.__name__}SuccessEnvelope",
        __base__=BaseModel,
        __doc__="Success or needs_review requires concrete data and error=null.",
        status=(Literal["success", "needs_review"], ...),
        data=(data_type, ...),
        error=(None, ...),
        meta=(Meta, ...),
    )


def _require_llm() -> None:
    if not get_settings().llm_configured:
        raise HTTPException(
            503,
            detail={
                "code": "LLM_NOT_CONFIGURED",
                "message": "未配置大模型，请联系管理员",
            },
        )


async def startup_event():
    global agents
    agents = {
        "quick_lesson_prep": QuickLessonPrepAgent(),
        "wisdom_transfer": WisdomTransferAgent(),
        "classroom_companion": ClassroomCompanionAgent(),
        "material": MaterialAgent(),
        "meeting_notes": MeetingNotesAgent(),
        "router": RouterAgent(),
    }


def _meta(started: float, model: str | None) -> dict:
    return {
        "model": model,
        "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "version": application_version(),
    }


def _model_used() -> str | None:
    from maitian_agent.llm import invoked_model

    state = invoked_model.get()
    return state.model if state is not None else None


def _legacy_failure(status: int, detail):
    raise HTTPException(status, detail=detail)


def _failure(status: int, code: str, message: str, started: float, *, legacy: bool, headers=None):
    if legacy:
        if status == 503:
            _legacy_failure(status, {"code": code, "message": message})
        _legacy_failure(status, "服务内部错误，请稍后重试" if status >= 500 else message)
    return JSONResponse(
        status_code=status,
        content={
            "status": "failed",
            "data": None,
            "error": {"code": code, "message": message},
            "meta": _meta(started, _model_used()),
        },
        headers=headers,
    )


def _http_error_code(status: int) -> str:
    return {
        400: "VALIDATION_ERROR",
        413: "UPLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
    }.get(status, "HTTP_ERROR" if status in (404, 405) else "INTERNAL_ERROR")


def _safe_message(code: str) -> str:
    return {
        "LLM_NOT_CONFIGURED": "未配置大模型，请联系管理员",
        "VALIDATION_ERROR": "请求参数无效，请检查后重试",
        "UPLOAD_TOO_LARGE": "上传内容过大，请上传大小限制以内的图片",
        "UNSUPPORTED_MEDIA_TYPE": "不支持的上传类型，请上传 PNG、JPG、JPEG 或 WebP 图片",
        "INTERNAL_ERROR": "服务内部错误，请稍后重试",
        "HTTP_ERROR": "请求路径或方法无效",
    }.get(code, "请求处理失败")


def _legacy_success(data, endpoint: str, status: str = "success"):
    if endpoint == "route":
        return {
            "success": True,
            "intent": data.intent,
            "agent": "RouterAgent",
            "info": {
                "intent": data.intent,
                "description": data.description,
                "available_intents": data.available_intents,
            },
        }
    result = {"success": True, "result": data.text}
    if status == "needs_review":
        result["status"] = status
    return result


def _v1_success(data, started: float, model: str | None, status="success"):
    return {
        "status": status,
        "data": data.model_dump(),
        "error": None,
        "meta": _meta(started, model),
    }


def _business(
    endpoint: str, request
) -> AgentResult[TextData] | AgentResult[RouteData] | AgentResult[WisdomData]:
    """Execute each existing business operation once and return typed AgentResult data."""
    if endpoint == "lesson-prep":
        return agents["quick_lesson_prep"].run(
            LessonPrepInput(
                subject=request.subject,
                grade=request.grade,
                topic=request.topic,
                rural_context=request.rural_context,
            )
        )
    if endpoint == "quiz":
        return agents["classroom_companion"].run(
            QuizInput(
                subject=request.subject,
                grade=request.grade,
                topic=request.topic,
                knowledge_points=request.knowledge_points,
                question_count=request.question_count,
                question_types=request.question_types,
            )
        )
    if endpoint == "material":
        return agents["material"].run(
            MaterialInput(
                subject=request.subject,
                grade=request.grade,
                topic=request.topic,
                knowledge_points=request.knowledge_points,
                rural_context=request.rural_context,
            )
        )
    if endpoint == "meeting-notes":
        return agents["meeting_notes"].run(MeetingNotesInput(transcript=request.transcript))
    if endpoint == "route":
        return agents["router"].run(RouteInput(user_input=request.user_input))
    raise ValueError("unknown endpoint")


@router.get("/", response_model=RootResponse)
async def root():
    return {"message": f"麦田智囊 API v{application_version()}", "status": "running"}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "agents": list(agents.keys()),
        "llm_configured": get_settings().llm_configured,
    }


def _add_json_route(path, endpoint, request_model, operation_id, *, legacy=False):
    async def handler(request):
        started = time.perf_counter()
        from maitian_agent.llm import InvocationState, invoked_model

        invocation_token = invoked_model.set(InvocationState())
        try:
            if endpoint == "route" and not request.user_input.strip():
                return _failure(
                    422,
                    "VALIDATION_ERROR",
                    _safe_message("VALIDATION_ERROR"),
                    started,
                    legacy=legacy,
                )
            _require_llm()
            result = await run_in_threadpool(_business, endpoint, request)
            if result.status == "failed":
                error = result.error
                code = error.code if error else "INTERNAL_ERROR"
                status_code = 422 if code == "VALIDATION_ERROR" else 500
                return _failure(
                    status_code,
                    code,
                    error.message if error else _safe_message(code),
                    started,
                    legacy=legacy,
                )
            if result.data is None:
                raise RuntimeError("agent returned no data")
            return (
                _legacy_success(result.data, endpoint)
                if legacy
                else _v1_success(result.data, started, _model_used())
            )
        except HTTPException as exc:
            code = (
                exc.detail.get("code", _http_error_code(exc.status_code))
                if isinstance(exc.detail, dict)
                else _http_error_code(exc.status_code)
            )
            message = (
                _safe_message(code)
                if code in ("VALIDATION_ERROR", "HTTP_ERROR", "INTERNAL_ERROR")
                else (
                    exc.detail.get("message", _safe_message(code))
                    if isinstance(exc.detail, dict)
                    else str(exc.detail)
                )
            )
            return _failure(exc.status_code, code, message, started, legacy=legacy)
        except LLMNotConfiguredError:
            return _failure(
                503,
                "LLM_NOT_CONFIGURED",
                _safe_message("LLM_NOT_CONFIGURED"),
                started,
                legacy=legacy,
            )
        except Exception:
            logger.error("API 请求处理失败（内部异常已脱敏）")
            return _failure(
                500, "INTERNAL_ERROR", _safe_message("INTERNAL_ERROR"), started, legacy=legacy
            )
        finally:
            invoked_model.reset(invocation_token)

    handler.__name__ = f"{'legacy_' if legacy else ''}{operation_id}"
    handler.__annotations__ = {"request": request_model, "return": dict}
    data_model = RouteData if endpoint == "route" else TextData
    response_model = (
        LegacyRouteResponse
        if legacy and endpoint == "route"
        else LegacyTextResponse
        if legacy
        else _success_envelope_for(data_model)
    )
    failed_response = {
        "description": "统一失败信封",
        "content": {"application/json": {"schema": FailedEnvelope.model_json_schema()}},
    }
    responses = (
        {
            422: {"description": "Legacy validation error"},
            500: {"description": "Legacy error"},
            503: {"description": "Legacy configuration error"},
        }
        if legacy
        else {code: failed_response for code in (400, 404, 405, 413, 415, 422, 500, 503)}
    )
    router.add_api_route(
        path,
        handler,
        methods=["POST"],
        operation_id=(
            f"legacy{operation_id[0].upper()}{operation_id[1:]}" if legacy else operation_id
        ),
        deprecated=legacy,
        response_model=response_model,
        responses=responses,
    )


_add_json_route("/api/lesson-prep", "lesson-prep", LessonPrepRequest, "lessonPrep", legacy=True)
_add_json_route("/api/quiz", "quiz", QuizRequest, "generateQuiz", legacy=True)
_add_json_route("/api/material", "material", MaterialRequest, "recommendMaterial", legacy=True)
_add_json_route(
    "/api/meeting-notes", "meeting-notes", MeetingNotesRequest, "processMeetingNotes", legacy=True
)
_add_json_route("/api/route", "route", RouteRequest, "routeIntent", legacy=True)
_add_json_route("/api/v1/lesson-prep", "lesson-prep", LessonPrepRequest, "lessonPrep")
_add_json_route("/api/v1/quiz", "quiz", QuizRequest, "generateQuiz")
_add_json_route("/api/v1/material", "material", MaterialRequest, "recommendMaterial")
_add_json_route(
    "/api/v1/meeting-notes", "meeting-notes", MeetingNotesRequest, "processMeetingNotes"
)
_add_json_route("/api/v1/route", "route", RouteRequest, "routeIntent")


async def _wisdom(request: Request, file: UploadFile, legacy: bool):
    started = time.perf_counter()
    from maitian_agent.llm import InvocationState, invoked_model

    invocation_token = invoked_model.set(InvocationState())
    image_path = None
    try:
        _require_llm()
        image_path = await save_image_upload(file, request.app.state.settings)
        result = await run_in_threadpool(
            agents["wisdom_transfer"].run, WisdomInput(image_path=str(image_path))
        )
        if not isinstance(result, AgentResult) or result.status == "failed" or result.data is None:
            raise RuntimeError("invalid OCR result")
        data, status = result.data, result.status
        used_model = _model_used()
        return (
            _legacy_success(data, "wisdom-transfer", status)
            if legacy
            else _v1_success(data, started, used_model, status)
        )
    except HTTPException as exc:
        code = (
            exc.detail.get("code", _http_error_code(exc.status_code))
            if isinstance(exc.detail, dict)
            else _http_error_code(exc.status_code)
        )
        message = (
            exc.detail.get("message", _safe_message(code))
            if isinstance(exc.detail, dict)
            else str(exc.detail)
        )
        return _failure(exc.status_code, code, message, started, legacy=legacy)
    except LLMNotConfiguredError:
        return _failure(
            503, "LLM_NOT_CONFIGURED", _safe_message("LLM_NOT_CONFIGURED"), started, legacy=legacy
        )
    except Exception:
        logger.error("智慧传承处理失败（内部异常已脱敏）")
        return _failure(
            500, "INTERNAL_ERROR", _safe_message("INTERNAL_ERROR"), started, legacy=legacy
        )
    finally:
        invoked_model.reset(invocation_token)
        if image_path is not None:
            try:
                delete_saved_upload(image_path, request.app.state.settings.upload_dir)
            except Exception:
                logger.exception("删除本次上传文件失败")


@router.post(
    "/api/wisdom-transfer",
    operation_id="legacyWisdomTransfer",
    deprecated=True,
    response_model=LegacyWisdomResponse,
    responses={
        413: {"description": "Legacy upload size error"},
        415: {"description": "Legacy media type error"},
        422: {"description": "Legacy validation error"},
        500: {"description": "Legacy internal error"},
        503: {"description": "Legacy configuration error"},
    },
)
async def legacy_wisdom_transfer(request: Request, file: UploadFile):
    return await _wisdom(request, file, True)


@router.post(
    "/api/v1/wisdom-transfer",
    operation_id="wisdomTransfer",
    response_model=_success_envelope_for(WisdomData),
    responses={
        413: {"description": "UPLOAD_TOO_LARGE"},
        415: {"description": "UNSUPPORTED_MEDIA_TYPE"},
        422: {"description": "VALIDATION_ERROR"},
        503: {"description": "LLM_NOT_CONFIGURED"},
        500: {"description": "INTERNAL_ERROR"},
    },
)
async def v1_wisdom_transfer(request: Request, file: UploadFile):
    return await _wisdom(request, file, False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stale_uploads(app.state.settings.upload_dir)
    await startup_event()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="麦田智囊API", version=application_version(), lifespan=lifespan)
    app.openapi = lambda: openapi_document(app)
    app.state.settings = settings
    app.add_middleware(UploadLimitMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in settings.allowed_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError):
        started = time.perf_counter()
        if get_route_path(request.scope).startswith("/api/v1/"):
            return _failure(
                422, "VALIDATION_ERROR", _safe_message("VALIDATION_ERROR"), started, legacy=False
            )
        return JSONResponse(status_code=422, content={"detail": "请求参数无效，请检查后重试"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        route_path = get_route_path(request.scope)
        headers = getattr(exc, "headers", None)
        if route_path.startswith("/api/v1/"):
            code = _http_error_code(exc.status_code)
            message = _safe_message(code)
            return _failure(
                exc.status_code, code, message, time.perf_counter(), legacy=False, headers=headers
            )
        if exc.status_code == 503 and isinstance(exc.detail, dict):
            return JSONResponse(status_code=503, content={"detail": exc.detail}, headers=headers)
        if exc.status_code >= 500:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": "服务内部错误，请稍后重试"},
                headers=headers,
            )
        detail = exc.detail if isinstance(exc.detail, str) else "请求无效，请检查后重试"
        return JSONResponse(
            status_code=exc.status_code, content={"detail": detail}, headers=headers
        )

    app.include_router(router)
    return app


app = create_app()
