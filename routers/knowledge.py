"""动态知识库 API 路由"""
from fastapi import APIRouter

from schemas.knowledge import (
    KnowledgeFetchRequest, KnowledgeFetchResponse,
    KnowledgeUpdateRequest, KnowledgeUpdateResponse,
    KnowledgeQueryRequest, KnowledgeQueryResponse,
)
from services.knowledge_service import knowledge_service

router = APIRouter(prefix="/api/knowledge", tags=["动态知识库"])


@router.post("/fetch", response_model=KnowledgeFetchResponse, summary="抓取网页内容并更新知识库")
async def fetch_url(req: KnowledgeFetchRequest):
    """
    从指定 URL 抓取网页内容，并可选择自动更新到知识库。

    - **url**: 要抓取的网页地址
    - **auto_update**: 是否自动更新到向量库（默认 True）
    """
    # 先抓取
    fetch_result = knowledge_service.fetch_url(req.url)
    if not fetch_result["success"]:
        return fetch_result

    response = {
        "success": True,
        "content": fetch_result.get("content", ""),
        "content_length": fetch_result.get("content_length", 0),
    }

    # 自动更新知识库
    if req.auto_update:
        update_result = knowledge_service.update_knowledge_base(
            fetch_result["content"]
        )
        response["update_result"] = update_result
        if not update_result["success"]:
            response["error"] = f"抓取成功但更新失败: {update_result['error']}"

    return response


@router.post("/update", response_model=KnowledgeUpdateResponse, summary="手动更新知识库")
async def update_knowledge(req: KnowledgeUpdateRequest):
    """
    将自定义文本内容添加到知识库中。

    - **text**: 要添加的文本内容
    """
    result = knowledge_service.update_knowledge_base(req.text)
    return result


@router.post("/query", response_model=KnowledgeQueryResponse, summary="查询动态知识库")
async def query_knowledge(req: KnowledgeQueryRequest):
    """
    基于已抓取并存储的知识库进行 RAG 问答。

    - **question**: 要查询的问题
    """
    result = knowledge_service.query(req.question)
    return result
