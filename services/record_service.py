"""
交互记录服务：异步写 PostgreSQL + 同步 Redis 缓存

写入链路（fire-and-forget，不阻塞主流程）：
    service 完成问答/创作 → submit_save(data)
      → async endpoint：asyncio.create_task(save_record)
      → Gradio 线程：daemon 线程 + asyncio.run(save_record)
      → save_record：await INSERT PG → await SET Redis 缓存（TTL 3600s）

读取链路：
    GET /api/records/{id}  单条：Redis 优先，未命中回源 PG 并回填
    GET /api/records       列表：直接走 PG（复合索引 + 分页）

容错降级：PostgreSQL / Redis 未启动时仅记日志，问答/创作主流程不受影响；
读取接口在存储不可用时由路由层返回 503。
"""
import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Optional

# 注意：redis.asyncio 在 initialize() 中惰性导入——
# 本地未安装 redis/psycopg 包或存储未启动时，仅记录模块降级，
# 不影响 qa_service/creative_service 的 import 链与服务启动。
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import DATABASE_URL, REDIS_URL
from services.db_models import Base, InteractionRecord

logger = logging.getLogger(__name__)

# Redis 键与缓存有效期
_RECORD_KEY_PREFIX = "rag:record:"
_RECORD_CACHE_TTL = 3600  # 秒

# 允许从 data 映射到 ORM 列的字段（白名单，防止无关字段写入报错）
_MODEL_FIELDS = {
    "record_type", "model", "user_input", "output",
    "sources", "extra", "status", "error_message",
}


class RecordStoreUnavailable(Exception):
    """记录存储（PostgreSQL）不可用，读取接口返回 503"""


class RecordService:
    """交互记录服务单例"""

    def __init__(self):
        self._engine = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._redis = None
        self._db_ready = False
        self._redis_ready = False
        self._initialized = False

    # ============ 初始化（main.py startup 调用，幂等） ============
    async def initialize(self) -> None:
        """创建连接池/客户端并建表；任何失败仅 warning，不影响服务启动"""
        if self._initialized:
            return
        self._initialized = True

        # PostgreSQL
        try:
            self._engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
            async with self._engine.connect() as conn:
                await conn.execute(func.now())  # 探测连接
            self._session_factory = async_sessionmaker(
                self._engine, expire_on_commit=False
            )
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._db_ready = True
            logger.info("✅ 交互记录服务已连接 PostgreSQL")
        except Exception as e:
            logger.warning(f"⚠️  无法连接 PostgreSQL（{DATABASE_URL.split('@')[-1]}），"
                           f"记录将不会落库，问答/创作不受影响: {e}")

        # Redis（惰性导入，未安装 redis 包时同样降级）
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self._redis.ping()
            self._redis_ready = True
            logger.info("✅ 交互记录服务已连接 Redis")
        except Exception as e:
            logger.warning(f"⚠️  无法连接 Redis（{REDIS_URL}），详情缓存不可用: {e}")

    # ============ 写入（异步落库 → 同步缓存） ============
    async def save_record(self, data: dict) -> None:
        """异步写 PG，成功后写 Redis 详情缓存；任何异常仅记录日志"""
        if not self._db_ready:
            logger.debug("PostgreSQL 不可用，跳过记录写入")
            return

        record_id: int | None = None
        created_at: datetime | None = None
        try:
            clean = {k: v for k, v in data.items() if k in _MODEL_FIELDS}
            async with self._session_factory() as session:  # type: ignore[union-attr]
                rec = InteractionRecord(**clean)
                session.add(rec)
                await session.commit()
                await session.refresh(rec)
                record_id = rec.id
                created_at = rec.created_at
        except Exception:
            logger.exception(
                "记录落库失败 (type=%s)", data.get("record_type", "unknown")
            )
            return

        # 同步 Redis 缓存（失败仅警告，不影响 PG 落库结果）
        if self._redis_ready and record_id is not None:
            try:
                payload = {**data, "id": record_id}
                if created_at is not None:
                    payload["created_at"] = created_at.isoformat()
                await self._redis.set(
                    f"{_RECORD_KEY_PREFIX}{record_id}",
                    json.dumps(payload, ensure_ascii=False, default=str),
                    ex=_RECORD_CACHE_TTL,
                )
            except Exception:
                logger.warning(
                    "Redis 缓存写入失败 (type=%s, id=%s)",
                    data.get("record_type", "unknown"), record_id,
                )

    # ============ 读取 ============
    async def get_record(self, record_id: int) -> dict | None:
        """单条详情：Redis 优先，未命中回源 PG 并回填缓存"""
        # 1) Redis 优先
        if self._redis_ready:
            try:
                raw = await self._redis.get(f"{_RECORD_KEY_PREFIX}{record_id}")
                if raw:
                    return json.loads(raw)
            except Exception:
                logger.warning("Redis 读取失败，回源 PostgreSQL (id=%s)", record_id)

        # 2) 回源 PostgreSQL
        if not self._db_ready:
            raise RecordStoreUnavailable
        try:
            async with self._session_factory() as session:  # type: ignore[union-attr]
                rec = await session.get(InteractionRecord, record_id)
            if rec is None:
                return None
            data = self._serialize(rec)
        except Exception:
            logger.exception("PostgreSQL 读取失败 (id=%s)", record_id)
            raise RecordStoreUnavailable from None

        # 3) 回填 Redis 缓存
        if self._redis_ready:
            try:
                await self._redis.set(
                    f"{_RECORD_KEY_PREFIX}{record_id}",
                    json.dumps(data, ensure_ascii=False, default=str),
                    ex=_RECORD_CACHE_TTL,
                )
            except Exception:
                pass
        return data

    async def list_records(
        self,
        record_type: str | None = None,
        keyword: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """列表分页查询（直接走 PG，复合索引 + 参数化查询）"""
        if not self._db_ready:
            raise RecordStoreUnavailable

        stmt = select(InteractionRecord)
        if record_type:
            stmt = stmt.where(InteractionRecord.record_type == record_type)
        if start_time is not None:
            stmt = stmt.where(InteractionRecord.created_at >= start_time)
        if end_time is not None:
            stmt = stmt.where(InteractionRecord.created_at <= end_time)
        if keyword:
            stmt = stmt.where(InteractionRecord.user_input.ilike(f"%{keyword}%"))

        try:
            async with self._session_factory() as session:  # type: ignore[union-attr]
                total = await session.scalar(
                    select(func.count()).select_from(stmt.subquery())
                )
                rows = await session.scalars(
                    stmt
                    .order_by(
                        InteractionRecord.created_at.desc(),
                        InteractionRecord.id.desc(),
                    )
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
                items = [self._serialize(r) for r in rows]
        except Exception:
            logger.exception("PostgreSQL 列表查询失败")
            raise RecordStoreUnavailable from None

        return {
            "total": total or 0,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    @staticmethod
    def _serialize(rec: InteractionRecord) -> dict:
        return {
            "id": rec.id,
            "record_type": rec.record_type,
            "model": rec.model,
            "user_input": rec.user_input,
            "output": rec.output,
            "sources": rec.sources,
            "extra": rec.extra,
            "status": rec.status,
            "error_message": rec.error_message,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }


# 全局单例
record_service = RecordService()


def submit_save(data: dict) -> None:
    """fire-and-forget 调度：有 running loop 用 create_task，否则 daemon 线程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop.create_task(record_service.save_record(data))
    else:
        threading.Thread(
            target=_run_save_in_thread, args=(data,), daemon=True
        ).start()


def _run_save_in_thread(data: dict) -> None:
    try:
        asyncio.run(record_service.save_record(data))
    except Exception:
        logger.exception("记录写入线程异常 (type=%s)", data.get("record_type"))
