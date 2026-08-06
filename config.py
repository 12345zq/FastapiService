"""
Fastapi_main 全局配置文件
所有 demo 应用的模型、路径等参数集中管理
"""
from pathlib import Path

# ============ 项目根目录 ============
BASE_DIR = Path(__file__).parent.resolve()

# ============ Ollama 服务 ============
OLLAMA_BASE_URL = "http://localhost:11434"

# ============ demo01: 知识库问答 ============
QA_MODEL = "deepseek-r1:latest"
QA_EMBED_MODEL = "bge-m3:latest"
QA_DATA_DIR = BASE_DIR / "data" / "text01"
QA_DB_DIR = BASE_DIR / "db" / "demo01"

# ============ demo02: 内容创作生成 ============
CREATIVE_MODEL = "deepseek-r1:latest"
CREATIVE_EMBED_MODEL = "bge-m3"
CREATIVE_DATA_DIR = BASE_DIR / "data" / "text02"
CREATIVE_DB_DIR = BASE_DIR / "db" / "demo02"

# ============ demo03: 多模态图像分析 ============
MM_MODEL = "llava:7b"
MM_EMBED_MODEL = "nomic-embed-text"
MM_DATA_DIR = BASE_DIR / "data" / "text03"
MM_DB_DIR = BASE_DIR / "db" / "demo03"

# ============ demo04: 动态知识库 ============
KNOWLEDGE_MODEL = "deepseek-r1:latest"
KNOWLEDGE_EMBED_MODEL = "nomic-embed-text"
KNOWLEDGE_DB_DIR = BASE_DIR / "db" / "demo04"

# ============ practice: BGE 本地模型 ============
BGE_MODEL_PATH = (BASE_DIR / "models_local" / "bge-small-zh-v1.5" / "bge-small-zh-v1.5").resolve()

# ============ practice: 数据目录 ============
MEDICAL_DATA_DIR = BASE_DIR / "practice_data" / "medical"
MOVIES_DATA_DIR = BASE_DIR / "practice_data" / "movies"

# ============ 上传文件保存目录 ============
UPLOAD_DIR = BASE_DIR / "uploads"
