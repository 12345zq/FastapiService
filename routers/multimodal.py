"""多模态图像分析 API 路由"""
from fastapi import APIRouter, UploadFile, File

from schemas.multimodal import MMAnalyzeResponse
from services.multimodal_service import multi_modal_service

router = APIRouter(prefix="/api/multimodal", tags=["多模态分析"])


@router.post("/analyze", response_model=MMAnalyzeResponse, summary="上传图像并生成分析报告")
async def analyze_image(file: UploadFile = File(..., description="要分析的图像文件（PNG/JPG）")):
    """
    上传一张图片，由 LLaVA 视觉模型描述图像内容，
    结合知识库生成综合分析报告。

    - **file**: 图像文件（支持 PNG/JPG/GIF 等格式）
    """
    if file.content_type and not file.content_type.startswith("image/"):
        return {"success": False, "error": "请上传有效的图像文件"}

    image_bytes = await file.read()
    if not image_bytes:
        return {"success": False, "error": "上传文件为空"}

    result = multi_modal_service.analyze_image(
        image_bytes=image_bytes,
        image_filename=file.filename or "uploaded.png",
    )
    return result
