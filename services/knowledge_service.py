"""
封装 demo04 的动态知识库管理服务
功能：网络抓取 + 增量向量库更新 + RAG 问答
模型：MFDoom/deepseek-r1-tool-calling:7b | nomic-embed-text
向量库：Chroma 持久化
"""
import logging
import os
import ollama
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config import KNOWLEDGE_MODEL, KNOWLEDGE_EMBED_MODEL, KNOWLEDGE_DB_DIR, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class KnowledgeService:
    """动态知识库管理服务（基于 demo04 逻辑）"""

    def __init__(self):
        self.vector_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.embeddings = OllamaEmbeddings(model=KNOWLEDGE_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self._initialized = False

    def initialize(self):
        """启动时加载已有向量库"""
        if self._initialized:
            return

        logger.info("KnowledgeService: 加载已有向量库...")
        self.vector_db = self._load_existing_db()
        self._initialized = True
        logger.info("KnowledgeService: 初始化完成")

    def _load_existing_db(self):
        if os.path.exists(str(KNOWLEDGE_DB_DIR)) and any(KNOWLEDGE_DB_DIR.iterdir()):
            try:
                return Chroma(
                    persist_directory=str(KNOWLEDGE_DB_DIR),
                    embedding_function=self.embeddings,
                )
            except Exception as e:
                logger.warning(f"KnowledgeService: 加载向量库失败 {e}")
        return None

    def fetch_url(self, url: str, timeout: int = 10) -> dict:
        """从指定 URL 抓取内容"""
        try:
            logger.info(f"KnowledgeService: 抓取 {url}")
            response = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            })
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, "html.parser")
            for s in soup(["script", "style"]):
                s.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if not text.strip():
                return {"success": False, "error": "抓取内容为空"}
            return {"success": True, "content": text, "content_length": len(text)}
        except requests.Timeout:
            return {"success": False, "error": f"请求超时 ({timeout}s)"}
        except Exception as e:
            logger.exception("KnowledgeService fetch_url 出错")
            return {"success": False, "error": str(e)}

    def update_knowledge_base(self, text: str) -> dict:
        """将文本增量更新到向量库"""
        if not text or not text.strip():
            return {"success": False, "error": "没有可更新的内容"}
        try:
            texts = self.text_splitter.split_text(text)
            if self.vector_db:
                self.vector_db.add_texts(texts)
            else:
                self.vector_db = Chroma.from_texts(
                    texts, self.embeddings, persist_directory=str(KNOWLEDGE_DB_DIR)
                )
            self.vector_db.persist()
            return {"success": True, "message": f"已添加 {len(texts)} 个文本块到知识库"}
        except Exception as e:
            logger.exception("KnowledgeService update 出错")
            return {"success": False, "error": str(e)}

    def query(self, question: str) -> dict:
        """基于动态知识库进行 RAG 问答"""
        if not self.vector_db:
            return {"success": False, "error": "知识库为空，请先抓取并更新内容"}
        try:
            docs = self.vector_db.similarity_search(question, k=10)
            context = "\n".join([d.page_content for d in docs])
            response = ollama.Client(host=OLLAMA_BASE_URL).generate(
                model=KNOWLEDGE_MODEL,
                prompt=f"基于以下上下文:\n{context}\n问题: {question}\n",
            )
            return {
                "success": True,
                "answer": response["response"],
                "source_count": len(docs),
            }
        except Exception as e:
            logger.exception("KnowledgeService query 出错")
            return {"success": False, "error": str(e)}


# 全局单例
knowledge_service = KnowledgeService()
