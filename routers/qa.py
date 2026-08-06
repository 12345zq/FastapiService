"""知识库问答 API 路由"""
from fastapi import APIRouter

from schemas.qa import QAAskRequest, QAAskResponse, QAReloadResponse
from services.qa_service import qa_service

router = APIRouter(prefix="/api/qa", tags=["知识库问答"])


@router.post("/ask", response_model=QAAskResponse, summary="RAG 知识库问答")
async def ask_question(req: QAAskRequest):
    """
    基于《大美安徽》知识库的 RAG 问答接口。

    - **question**: 要提问的问题（如：黄山有哪些著名景点？）
    """
    result = qa_service.ask(req.question)
    return result


@router.post("/reload", response_model=QAReloadResponse, summary="重新构建向量库")
async def reload_database():
    """
    重新加载文档并构建向量库。
    
    当文档数据有更新时调用此接口刷新知识库。
    """
    result = qa_service.reload()
    return result
