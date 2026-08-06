"""内容创作 API 路由"""
from fastapi import APIRouter

from schemas.creative import CreativeGenerateRequest, CreativeGenerateResponse
from services.creative_service import creative_service

router = APIRouter(prefix="/api/creative", tags=["内容创作"])


@router.post("/generate", response_model=CreativeGenerateResponse, summary="生成创作内容")
async def generate_content(req: CreativeGenerateRequest):
    """
    基于创作技巧知识库的内容生成接口。

    - **genre**: 创作类型（如：故事、诗歌、文案等）
    - **requirements**: 具体要求描述
    """
    result = creative_service.generate(req.genre, req.requirements)
    return result
