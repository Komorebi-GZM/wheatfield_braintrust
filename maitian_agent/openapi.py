"""Deterministic, offline OpenAPI contract export."""

import json
import sys
import tomllib
from pathlib import Path

INTENTS = [
    "quick_lesson_prep",
    "wisdom_transfer",
    "classroom_companion",
    "material_recommend",
    "meeting_notes",
    "general_chat",
]
ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

REQUESTS = {
    "lesson-prep": {
        "type": "object",
        "required": ["subject", "grade", "topic"],
        "properties": {
            "subject": {"type": "string"},
            "grade": {"type": "string"},
            "topic": {"type": "string"},
            "rural_context": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": ""},
        },
    },
    "quiz": {
        "type": "object",
        "required": ["subject", "grade", "topic"],
        "properties": {
            "subject": {"type": "string"},
            "grade": {"type": "string"},
            "topic": {"type": "string"},
            "knowledge_points": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": ""},
            "question_count": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 5},
            "question_types": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": "选择题、填空题",
            },
        },
    },
    "material": {
        "type": "object",
        "required": ["subject", "grade", "topic"],
        "properties": {
            "subject": {"type": "string"},
            "grade": {"type": "string"},
            "topic": {"type": "string"},
            "knowledge_points": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": ""},
            "rural_context": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": ""},
        },
    },
    "meeting-notes": {
        "type": "object",
        "required": ["transcript"],
        "properties": {"transcript": {"type": "string"}},
    },
    "route": {
        "type": "object",
        "required": ["user_input"],
        "properties": {"user_input": {"type": "string"}},
    },
}


def openapi_document() -> dict:
    schemas = {
        "Meta": {
            "type": "object",
            "required": ["model", "duration_ms", "version"],
            "properties": {
                "model": {"type": ["string", "null"]},
                "duration_ms": {"type": "integer", "minimum": 0},
                "version": {"type": "string", "minLength": 1},
            },
        },
        "Error": {
            "type": "object",
            "required": ["code", "message"],
            "properties": {"code": {"type": "string"}, "message": {"type": "string"}},
        },
        "TextData": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "WisdomData": {
            "type": "object",
            "required": ["text", "review_reasons"],
            "properties": {
                "text": {"type": "string"},
                "review_reasons": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["OCR_REQUIRES_REVIEW", "OCR_NO_TEXT"]},
                },
            },
        },
        "RouteData": {
            "type": "object",
            "required": ["intent", "description", "available_intents"],
            "properties": {
                "intent": {"type": "string", "enum": INTENTS},
                "description": {"type": "string"},
                "available_intents": {
                    "type": "array",
                    "items": {"type": "string", "enum": INTENTS},
                },
            },
        },
        "SuccessText": {
            "type": "object",
            "description": "For success or needs_review, data is required and error is null.",
            "required": ["status", "data", "error", "meta"],
            "properties": {
                "status": {"type": "string", "enum": ["success", "needs_review"]},
                "data": {"$ref": "#/components/schemas/TextData"},
                "error": {"type": "null"},
                "meta": {"$ref": "#/components/schemas/Meta"},
            },
        },
        "SuccessWisdom": {
            "type": "object",
            "description": "For success or needs_review, data is required and error is null.",
            "required": ["status", "data", "error", "meta"],
            "properties": {
                "status": {"type": "string", "enum": ["success", "needs_review"]},
                "data": {"$ref": "#/components/schemas/WisdomData"},
                "error": {"type": "null"},
                "meta": {"$ref": "#/components/schemas/Meta"},
            },
        },
        "SuccessRoute": {
            "type": "object",
            "description": "For success or needs_review, data is required and error is null.",
            "required": ["status", "data", "error", "meta"],
            "properties": {
                "status": {"type": "string", "enum": ["success", "needs_review"]},
                "data": {"$ref": "#/components/schemas/RouteData"},
                "error": {"type": "null"},
                "meta": {"$ref": "#/components/schemas/Meta"},
            },
        },
        "Failed": {
            "type": "object",
            "description": "For failed, data is null and error is required.",
            "required": ["status", "data", "error", "meta"],
            "properties": {
                "status": {"const": "failed"},
                "data": {"type": "null"},
                "error": {"$ref": "#/components/schemas/Error"},
                "meta": {"$ref": "#/components/schemas/Meta"},
            },
        },
    }
    common_responses = {
        "422": {
            "description": "VALIDATION_ERROR",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Failed"},
                    "example": {
                        "status": "failed",
                        "data": None,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "请求参数无效，请检查后重试",
                        },
                        "meta": {"model": None, "duration_ms": 0, "version": VERSION},
                    },
                }
            },
        },
        "503": {
            "description": "LLM_NOT_CONFIGURED",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Failed"},
                    "example": {
                        "status": "failed",
                        "data": None,
                        "error": {
                            "code": "LLM_NOT_CONFIGURED",
                            "message": "未配置大模型，请联系管理员",
                        },
                        "meta": {"model": None, "duration_ms": 0, "version": VERSION},
                    },
                }
            },
        },
        "500": {
            "description": "INTERNAL_ERROR",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/Failed"},
                    "example": {
                        "status": "failed",
                        "data": None,
                        "error": {"code": "INTERNAL_ERROR", "message": "服务内部错误，请稍后重试"},
                        "meta": {"model": None, "duration_ms": 0, "version": VERSION},
                    },
                }
            },
        },
    }
    endpoint_ids = {
        "lesson-prep": "lessonPrep",
        "wisdom-transfer": "wisdomTransfer",
        "quiz": "generateQuiz",
        "material": "recommendMaterial",
        "meeting-notes": "processMeetingNotes",
        "route": "routeIntent",
    }
    paths = {
        "/": {"get": {"operationId": "root", "responses": {"200": {"description": "Root status"}}}},
        "/health": {
            "get": {
                "operationId": "healthCheck",
                "responses": {"200": {"description": "Health status"}},
            }
        },
    }
    for suffix, operation_id in endpoint_ids.items():
        legacy_path, v1_path = f"/api/{suffix}", f"/api/v1/{suffix}"
        is_upload = suffix == "wisdom-transfer"
        request_body = (
            {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {"file": {"type": "string", "format": "binary"}},
                        }
                    }
                },
            }
            if is_upload
            else {"required": True, "content": {"application/json": {"schema": REQUESTS[suffix]}}}
        )
        legacy_id = "legacy" + operation_id[0].upper() + operation_id[1:]
        if suffix == "route":
            legacy_response = {
                "type": "object",
                "required": ["success", "intent", "agent", "info"],
                "properties": {
                    "success": {"const": True},
                    "intent": {"type": "string", "enum": INTENTS},
                    "agent": {"type": "string"},
                    "info": {"$ref": "#/components/schemas/RouteData"},
                },
            }
        else:
            legacy_response = {
                "type": "object",
                "required": ["success", "result"],
                "properties": {
                    "success": {"const": True},
                    "result": {"type": "string"},
                },
            }
            if is_upload:
                legacy_response["properties"]["status"] = {
                    "type": "string",
                    "enum": ["needs_review"],
                }
        paths[legacy_path] = {
            "post": {
                "operationId": legacy_id,
                "deprecated": True,
                "requestBody": request_body,
                "responses": {
                    "200": {
                        "description": "Legacy success",
                        "content": {"application/json": {"schema": legacy_response}},
                    },
                    "422": {"description": "Legacy validation error"},
                    "500": {"description": "Legacy internal error"},
                    "503": {"description": "Legacy configuration error"},
                },
            }
        }
        success_name = (
            "SuccessWisdom" if is_upload else "SuccessRoute" if suffix == "route" else "SuccessText"
        )
        success_data = (
            {"text": "", "review_reasons": ["OCR_NO_TEXT"]}
            if is_upload
            else (
                {"intent": "general_chat", "description": "通用对话", "available_intents": INTENTS}
                if suffix == "route"
                else {"text": "生成内容"}
            )
        )
        responses = {
            "200": {
                "description": "Success or needs_review",
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{success_name}"},
                        "example": {
                            "status": "needs_review" if is_upload else "success",
                            "data": success_data,
                            "error": None,
                            "meta": {"model": None, "duration_ms": 0, "version": VERSION},
                        },
                    }
                },
            },
            **common_responses,
        }
        if is_upload:
            for code, label in [(413, "UPLOAD_TOO_LARGE"), (415, "UNSUPPORTED_MEDIA_TYPE")]:
                responses[str(code)] = {
                    "description": label,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Failed"},
                            "example": {
                                "status": "failed",
                                "data": None,
                                "error": {
                                    "code": label,
                                    "message": (
                                        "上传内容过大，请上传大小限制以内的图片"
                                        if code == 413
                                        else "不支持的上传类型，请上传 PNG、JPG、JPEG 或 WebP 图片"
                                    ),
                                },
                                "meta": {"model": None, "duration_ms": 0, "version": VERSION},
                            },
                        }
                    },
                }
        paths[v1_path] = {
            "post": {
                "operationId": operation_id,
                "requestBody": request_body,
                "responses": responses,
            }
        }
    return {
        "openapi": "3.1.0",
        "info": {"title": "麦田智囊 API", "version": VERSION},
        "paths": paths,
        "components": {"schemas": schemas},
    }


def export_openapi(path: str | Path = "openapi.json") -> None:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    target.write_text(
        json.dumps(openapi_document(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    export_openapi(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
