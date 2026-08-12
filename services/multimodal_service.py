"""
封装 demo03 的多模态 RAG 图像分析服务
文档：text03/大熊猫 TXT + PNG
模型：llava:7b（视觉理解）| nomic-embed-text（嵌入）
向量库：Chroma 持久化
"""
import logging
import os
import ollama
from pathlib import Path
from PIL import Image
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from config import MM_MODEL, MM_EMBED_MODEL, MM_DATA_DIR, MM_DB_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)


class MultiModalService:
    """多模态图像分析服务（基于 demo03 逻辑）"""

    def __init__(self):
        self.text_db = None
        self._initialized = False

    def initialize(self):
        """启动时构建/加载文本向量库"""
        if self._initialized:
            return

        logger.info("MultiModalService: 初始化文本向量库...")
        self.text_db = self._prepare_text_db()
        self._initialized = True
        logger.info("MultiModalService: 初始化完成")

    def _prepare_text_db(self):
        embeddings = OllamaEmbeddings(model=MM_EMBED_MODEL)
        if MM_DB_DIR.exists() and any(MM_DB_DIR.iterdir()):
            logger.info(f"MultiModalService: 加载已有向量库 {MM_DB_DIR}")
            return Chroma(
                persist_directory=str(MM_DB_DIR),
                embedding_function=embeddings,
            )
        loader = DirectoryLoader(
            str(MM_DATA_DIR), glob="**/*.txt",
            loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        logger.info("MultiModalService: 首次构建向量库...")
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(MM_DB_DIR),
        )

    def analyze_image(self, image_path: str = None, image_bytes: bytes = None,
                      image_filename: str = "uploaded.png") -> dict:
        """分析图像并生成综合报告"""
        if not self._initialized:
            return {"success": False, "error": "服务未初始化，请先调用 initialize()"}

        try:
            # 确定图像路径
            if image_bytes:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                save_path = UPLOAD_DIR / image_filename
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                img_path = str(save_path)
            elif image_path:
                img_path = image_path
            else:
                return {"success": False, "error": "请提供图像路径或上传图像"}

            if not os.path.exists(img_path):
                return {"success": False, "error": f"图像文件不存在: {img_path}"}

            # 预处理图像
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            temp_path = UPLOAD_DIR / "temp.jpg"
            img.save(str(temp_path))

            # 使用 LLaVA 描述图像
            logger.info(f"MultiModalService: 使用 {MM_MODEL} 描述图像...")
            response = ollama.generate(
                model=MM_MODEL,
                prompt="请描述这张图片内容（中文）: ",
                images=[img_path],
            )
            description = response["response"]

            # 检索相关文本知识
            docs = self.text_db.similarity_search(description, k=1)
            context = "\n".join([d.page_content for d in docs])

            # 生成综合分析报告
            final_response = ollama.generate(
                model=MM_MODEL,
                prompt=f"根据以下信息生成一份综合分析报告（中文）:\n\n"
                       f"图像描述: {description}\n\n"
                       f"相关知识: {context}\n\n ",
            )

            return {
                "success": True,
                "image_description": description,
                "related_info": [d.metadata for d in docs],
                "final_report": final_response["response"],
            }
        except Exception as e:
            logger.exception("MultiModalService analyze_image 出错")
            return {"success": False, "error": str(e)}


# 全局单例
multi_modal_service = MultiModalService()
