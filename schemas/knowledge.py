"""动态知识库接口 schema"""
from pydantic import BaseModel, Field


class KnowledgeFetchRequest(BaseModel):
    url: str = Field(..., description="要抓取的网页 URL", min_length=5, max_length=2048)
    auto_update: bool = Field(default=True, description="抓取后是否自动更新到知识库")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/article",
                "auto_update": True,
            }
        }


class KnowledgeUpdateRequest(BaseModel):
    text: str = Field(..., description="要添加到知识库的文本内容",
                      min_length=1, max_length=100000)


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(..., description="要查询的问题",
                          min_length=1, max_length=2000)


class KnowledgeFetchResponse(BaseModel):
    success: bool
    content: str | None = None
    content_length: int | None = None
    update_result: dict | None = None
    error: str | None = None


class KnowledgeUpdateResponse(BaseModel):
    success: bool
    message: str | None = None
    error: str | None = None


class KnowledgeQueryResponse(BaseModel):
    success: bool
    answer: str | None = None
    source_count: int | None = None
    error: str | None = None
