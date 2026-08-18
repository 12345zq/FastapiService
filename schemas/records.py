"""交互记录查询接口的 Pydantic 模型"""
from typing import Any, Optional

from pydantic import BaseModel


class RecordItem(BaseModel):
    """单条交互记录"""
    id: int
    record_type: str
    model: Optional[str] = None
    user_input: Optional[str] = None
    output: Optional[str] = None
    sources: Optional[Any] = None
    extra: Optional[Any] = None
    status: str = "success"
    error_message: Optional[str] = None
    created_at: Optional[str] = None


class RecordListResponse(BaseModel):
    """记录列表分页响应"""
    total: int
    page: int
    page_size: int
    items: list[RecordItem]


class RecordDetailResponse(BaseModel):
    """单条记录详情响应"""
    record: RecordItem
