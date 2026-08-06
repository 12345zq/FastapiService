"""知识库问答接口 schema"""
from pydantic import BaseModel, Field


class QAAskRequest(BaseModel):
    question: str = Field(..., description="要提问的问题", min_length=1, max_length=2000)

    class Config:
        json_schema_extra = {"example": {"question": "黄山有哪些著名景点？"}}


class SourceInfo(BaseModel):
    index: int = Field(..., description="参考序号")
    source: str = Field(..., description="来源文件名")
    content_preview: str = Field(..., description="内容预览（前100字符）")


class QAAskResponse(BaseModel):
    success: bool
    answer: str | None = None
    sources: list | None = None
    error: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "根据上下文，黄山最著名的景点包括...",
                "sources": [
                    {"index": 1, "source": "text01/黄山.md", "content_preview": "黄山位于..."}
                ],
                "error": None,
            }
        }


class QAReloadResponse(BaseModel):
    success: bool
    message: str | None = None
    error: str | None = None
