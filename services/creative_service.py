"""
封装 demo02 的内容创作生成服务
文档：text02/创作技巧 TXT
模型：deepseek-r1:latest | bge-m3
向量库：Chroma 持久化
"""
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import CREATIVE_MODEL, CREATIVE_EMBED_MODEL, CREATIVE_DATA_DIR, CREATIVE_DB_DIR

logger = logging.getLogger(__name__)


class CreativeService:
    """内容创作生成服务（基于 demo02 逻辑）"""

    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.retriever = None
        self._initialized = False

    def initialize(self):
        """启动时加载/构建向量库和生成管道"""
        if self._initialized:
            return

        logger.info("CreativeService: 加载文档...")
        docs = self._load_documents()

        logger.info("CreativeService: 构建/加载向量库...")
        self.vector_store = self._load_or_create_vectorstore(docs)

        logger.info("CreativeService: 初始化生成管道...")
        self._setup_chain()

        self._initialized = True
        logger.info("CreativeService: 初始化完成")

    def _load_documents(self):
        loader = DirectoryLoader(
            str(CREATIVE_DATA_DIR), glob="**/*.txt",
            loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=30
        )
        return text_splitter.split_documents(documents)

    def _load_or_create_vectorstore(self, docs):
        embeddings = OllamaEmbeddings(model=CREATIVE_EMBED_MODEL)
        if CREATIVE_DB_DIR.exists() and any(CREATIVE_DB_DIR.iterdir()):
            logger.info(f"CreativeService: 加载已有向量库 {CREATIVE_DB_DIR}")
            return Chroma(
                persist_directory=str(CREATIVE_DB_DIR),
                embedding_function=embeddings,
            )
        logger.info("CreativeService: 首次构建向量库...")
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(CREATIVE_DB_DIR),
        )

    def _setup_chain(self):
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        prompt_template = """基于以下上下文和创作技巧，生成{genre}内容：
上下文：{context}
要求：{requirements}
生成内容："""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["genre", "requirements", "context"],
        )

        def retrieve_and_build_input(input_dict):
            genre = input_dict["genre"]
            docs = self.retriever.invoke(f"{genre}创作技巧")
            input_dict["context"] = "\n\n".join([d.page_content for d in docs])
            return input_dict

        ollama_llm = ChatOllama(model=CREATIVE_MODEL)
        self.chain = (
            RunnableLambda(retrieve_and_build_input)
            | prompt
            | ollama_llm
            | StrOutputParser()
        )

    def generate(self, genre: str, requirements: str) -> dict:
        """生成创作内容，返回结果和参考来源"""
        if not self._initialized:
            return {"success": False, "error": "服务未初始化，请先调用 initialize()"}
        try:
            result = self.chain.invoke({
                "genre": genre,
                "requirements": requirements,
            })
            docs = self.retriever.invoke(f"{genre}创作技巧")
            sources = [
                {"index": i + 1, "source": doc.metadata.get("source", "未知"),
                 "content_preview": doc.page_content[:100]}
                for i, doc in enumerate(docs)
            ]
            return {"success": True, "result": result, "sources": sources}
        except Exception as e:
            logger.exception("CreativeService generate 出错")
            return {"success": False, "error": str(e)}


# 全局单例
creative_service = CreativeService()
