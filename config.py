"""
Fastapi_main 全局配置文件
所有 demo 应用的模型、路径等参数集中管理
"""
import os
from pathlib import Path

# ============ 项目根目录 ============
BASE_DIR = Path(__file__).parent.resolve()

# ============ Ollama 服务 ============
# Docker 部署时通过环境变量 OLLAMA_BASE_URL 覆盖（如 http://host.docker.internal:11434）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ============ OpenAI 兼容 Chat Completions API ============
# 默认派生自 OLLAMA_BASE_URL + "/v1"（本地 Ollama 的 OpenAI 兼容端点，只需 ollama serve）
# 可独立覆盖以切换到任意 OpenAI 兼容服务（如第三方云服务）
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", f"{OLLAMA_BASE_URL}/v1")
# 本地 Ollama 忽略 key 内容，仅要求非空占位；切第三方服务时改为真实 key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# ============ demo01: 知识库问答 ============
QA_MODEL = "MFDoom/deepseek-r1-tool-calling:7b"
QA_EMBED_MODEL = "bge-m3:latest"
QA_DATA_DIR = BASE_DIR / "data" / "text01"
QA_DB_DIR = BASE_DIR / "db" / "demo01"

# ============ demo02: 内容创作生成 ============
CREATIVE_MODEL = "MFDoom/deepseek-r1-tool-calling:7b"
CREATIVE_EMBED_MODEL = "bge-m3"
CREATIVE_DATA_DIR = BASE_DIR / "data" / "text02"
CREATIVE_DB_DIR = BASE_DIR / "db" / "demo02"

# ============ demo03: 多模态图像分析 ============
MM_MODEL = "llava:7b"
MM_EMBED_MODEL = "nomic-embed-text"
MM_DATA_DIR = BASE_DIR / "data" / "text03"
MM_DB_DIR = BASE_DIR / "db" / "demo03"

# ============ demo04: 动态知识库 ============
KNOWLEDGE_MODEL = "MFDoom/deepseek-r1-tool-calling:7b"
KNOWLEDGE_EMBED_MODEL = "nomic-embed-text"
KNOWLEDGE_DB_DIR = BASE_DIR / "db" / "demo04"

# ============ practice: BGE 本地模型 ============
BGE_MODEL_PATH = (BASE_DIR / "models_local" / "bge-small-zh-v1.5" / "bge-small-zh-v1.5").resolve()

# ============ practice: 数据目录 ============
DEMO_DATA_DIR = BASE_DIR / "practice_data" / "demo"
MOVIES_DATA_DIR = BASE_DIR / "practice_data" / "movies"

# ============ 上传文件保存目录 ============
UPLOAD_DIR = BASE_DIR / "uploads"
