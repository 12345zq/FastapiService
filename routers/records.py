"""交互记录查询 API 路由"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from schemas.records import RecordDetailResponse, RecordListResponse
from services.record_service import RecordStoreUnavailable, record_service

router = APIRouter(prefix="/api/records", tags=["交互记录"])


def _parse_datetime(value: str | None) -> datetime | None:
    """解析 ISO 格式时间字符串，非法时返回 422"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"无效的时间格式: {value}")


@router.get("", response_model=RecordListResponse, summary="查询交互记录列表")
async def list_records(
    record_type: str | None = Query(None, description="记录类型：qa / creative"),
    keyword: str | None = Query(None, description="关键词（模糊匹配用户输入）"),
    start_time: str | None = Query(
        None, description="起始时间（ISO 格式，如 2026-08-18T00:00:00）"
    ),
    end_time: str | None = Query(None, description="结束时间（ISO 格式）"),
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量（最大 100）"),
):
    """
    分页查询问答/创作记录。

    支持按 **record_type**（qa / creative）、**keyword**（匹配用户输入）、
    **start_time** / **end_time**（ISO 时间范围）过滤，按创建时间倒序返回。
    """
    try:
        return await record_service.list_records(
            record_type=record_type,
            keyword=keyword,
            start_time=_parse_datetime(start_time),
            end_time=_parse_datetime(end_time),
            page=page,
            page_size=page_size,
        )
    except RecordStoreUnavailable:
        raise HTTPException(status_code=503, detail="记录存储不可用（PostgreSQL 未连接）")


@router.get("/{record_id}", response_model=RecordDetailResponse, summary="查询单条记录详情")
async def get_record(record_id: int):
    """
    查询单条记录详情。

    Redis 缓存优先，未命中时回源 PostgreSQL 并回填缓存。
    """
    try:
        record = await record_service.get_record(record_id)
    except RecordStoreUnavailable:
        raise HTTPException(status_code=503, detail="记录存储不可用（PostgreSQL 未连接）")
    if record is None:
        raise HTTPException(status_code=404, detail=f"记录不存在: {record_id}")
    return {"record": record}
