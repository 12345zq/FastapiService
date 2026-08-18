"""
交互记录 ORM 模型

统一记录表 interaction_records：主字段 + JSONB 扩展字段，
覆盖 qa / creative 两类记录，结构可扩展至 multimodal / knowledge。
"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InteractionRecord(Base):
    """问答/创作交互记录"""

    __tablename__ = "interaction_records"
    __table_args__ = (
        # 列表查询最常用过滤组合：类型 + 时间范围（分页 ORDER BY created_at DESC）
        Index("ix_records_type_created", "record_type", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    record_type: Mapped[str] = mapped_column(String(20), nullable=False)  # qa | creative
    model: Mapped[str | None] = mapped_column(String(100))                 # 使用的模型
    user_input: Mapped[str | None] = mapped_column(Text)                   # 问题 / genre+requirements
    output: Mapped[str | None] = mapped_column(Text)                       # answer / result
    sources: Mapped[list | None] = mapped_column(JSONB)                    # 参考来源列表
    extra: Mapped[dict | None] = mapped_column(JSONB)                      # genre、requirements、source_count 等
    status: Mapped[str] = mapped_column(String(20), default="success")     # success | failed
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
