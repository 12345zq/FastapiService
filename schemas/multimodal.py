"""多模态图像分析接口 schema"""
from pydantic import BaseModel, Field


class MMAnalyzeResponse(BaseModel):
    success: bool
    image_description: str | None = None
    related_info: list | None = None
    final_report: str | None = None
    error: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "image_description": "图片中有一只大熊猫在吃竹子...",
                "related_info": [{"source": "text03/daxiongmao.txt"}],
                "final_report": "综合图像描述和知识库信息...",
                "error": None,
            }
        }
