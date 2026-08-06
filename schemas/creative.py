"""内容创作接口 schema"""
from pydantic import BaseModel, Field


class CreativeGenerateRequest(BaseModel):
    genre: str = Field(..., description="创作类型，如：故事、诗歌、文案等",
                       min_length=1, max_length=100)
    requirements: str = Field(..., description="具体要求",
                              min_length=1, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "genre": "故事",
                "requirements": "关于一个勇敢的小女孩在森林中冒险的故事，需要有趣且富有教育意义",
            }
        }


class CreativeGenerateResponse(BaseModel):
    success: bool
    result: str | None = None
    sources: list | None = None
    error: str | None = None
