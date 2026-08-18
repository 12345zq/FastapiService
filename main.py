"""
Fastapi_main 主入口

统一启动 FastAPI 服务 + Gradio 演示页面

启动方式：
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API 文档：
    http://localhost:8000/docs      Swagger UI
    http://localhost:8000/redoc     ReDoc

Gradio 演示：
    http://localhost:8000/gradio/query    医疗问答
"""
import logging
import sys
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import OLLAMA_BASE_URL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============ 创建 FastAPI 应用 ============
app = FastAPI(
    title="RAG 统一服务平台",
    description="""
## 功能模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 📚 知识库问答 | `/api/qa` | 基于《大美安徽》的 RAG 问答 |
| ✍️ 内容创作 | `/api/creative` | 基于创作技巧的内容生成 |
| 🖼️ 多模态分析 | `/api/multimodal` | 上传图片 + 知识库综合分析 |
| 🔄 动态知识库 | `/api/knowledge` | 网页抓取 + 增量更新 + 问答 |
| 🏥 智能问答 | `/gradio/query` | 知识库 Gradio 演示 |
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 注册 API 路由 ============
from routers import qa, creative, multimodal, knowledge

app.include_router(qa.router)
app.include_router(creative.router)
app.include_router(multimodal.router)
app.include_router(knowledge.router)


# ============ 启动事件：初始化各服务 ============
@app.on_event("startup")
async def startup_event():
    """服务器启动时初始化所有 RAG 服务"""
    import ollama

    # 检查 Ollama 服务
    try:
        ollama.Client(host=OLLAMA_BASE_URL).list()
        logger.info("Ollama 服务连接正常")
    except Exception:
        logger.warning(
            "⚠️  无法连接到 Ollama 服务！请先执行：ollama serve\n"
            "    API 接口将返回错误，但服务仍会启动"
        )

    # 初始化问答服务
    try:
        from services.qa_service import qa_service
        qa_service.initialize()
        logger.info("✅ 知识库问答服务已就绪")
    except Exception as e:
        logger.error(f"❌ 知识库问答服务初始化失败: {e}")

    # 初始化创作服务
    try:
        from services.creative_service import creative_service
        creative_service.initialize()
        logger.info("✅ 内容创作服务已就绪")
    except Exception as e:
        logger.error(f"❌ 内容创作服务初始化失败: {e}")

    # 初始化多模态服务
    try:
        from services.multimodal_service import multi_modal_service
        multi_modal_service.initialize()
        logger.info("✅ 多模态分析服务已就绪")
    except Exception as e:
        logger.error(f"❌ 多模态分析服务初始化失败: {e}")

    # 初始化知识库服务
    try:
        from services.knowledge_service import knowledge_service
        knowledge_service.initialize()
        logger.info("✅ 动态知识库服务已就绪")
    except Exception as e:
        logger.error(f"❌ 动态知识库服务初始化失败: {e}")

    logger.info("=" * 50)
    logger.info("🚀 RAG 统一服务平台已启动")
    logger.info("   API 文档: http://localhost:8000/docs")
    logger.info("   问答 API: http://localhost:8000/api/qa/ask")
    logger.info("   创作 API: http://localhost:8000/api/creative/generate")
    logger.info("   多模态 API: http://localhost:8000/api/multimodal/analyze")
    logger.info("   知识库 API: http://localhost:8000/api/knowledge/fetch")
    logger.info("   问答演示: http://localhost:8000/gradio/query")
    logger.info("=" * 50)


@app.get("/", summary="服务状态")
async def root():
    """根路径，返回服务状态"""
    return {
        "service": "RAG 统一服务平台",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/api",
        "modules": {
            "qa": "/api/qa",
            "creative": "/api/creative",
            "multimodal": "/api/multimodal",
            "knowledge": "/api/knowledge",
        },
        "gradio_demos": {
            "demo_qa": "/gradio/query"
        },
    }

# ============ 挂载 Gradio 应用 ============
try:
    from gradio_apps.demo_qa import create_demo_app
    demo_app = create_demo_app()
    app = gr.mount_gradio_app(app, demo_app, path="/gradio/query")
    logger.info("✅ Gradio 智能问答界面已挂载")
except Exception as e:
    logger.warning(f"⚠️  Gradio 智能问答界面加载失败: {e}")

# ============ 直接运行 ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
