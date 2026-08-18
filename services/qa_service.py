"""
封装 demo01 的 RAG 知识库问答服务
文档：text01/《大美安徽》Markdown 文档
模型：MFDoom/deepseek-r1-tool-calling:7b | bge-m3:latest
向量库：Chroma 持久化
"""
import logging
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import QA_MODEL, QA_EMBED_MODEL, QA_DATA_DIR, QA_DB_DIR, OPENAI_API_BASE, OPENAI_API_KEY

logger = logging.getLogger(__name__)


class QAService:
    """RAG 知识库问答服务（基于 demo01 逻辑）"""

    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self._initialized = False

    def initialize(self):
        """启动时加载/构建向量库和问答管道"""
        if self._initialized:
            return

        logger.info("QAService: 加载文档...")
        docs = self._load_documents()

        logger.info("QAService: 构建/加载向量库...")
        self.vectorstore = self._load_or_create_vectorstore(docs)

        logger.info("QAService: 初始化问答管道...")
        self._setup_qa_chain()

        self._initialized = True
        logger.info("QAService: 初始化完成")

    def _load_documents(self):
        loader = DirectoryLoader(
            str(QA_DATA_DIR), glob="**/*.md",
            loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        return text_splitter.split_documents(documents)

    def _load_or_create_vectorstore(self, docs):
        embeddings = OpenAIEmbeddings(model=QA_EMBED_MODEL, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_API_BASE, check_embedding_ctx_length=False)
        if QA_DB_DIR.exists() and any(QA_DB_DIR.iterdir()):
            logger.info(f"QAService: 加载已有向量库 {QA_DB_DIR}")
            return Chroma(
                persist_directory=str(QA_DB_DIR),
                embedding_function=embeddings,
            )
        logger.info("QAService: 首次构建向量库...")
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(QA_DB_DIR),
        )

    def _setup_qa_chain(self):
        llm = ChatOpenAI(model_name=QA_MODEL, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY, temperature=0)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        def _format_docs(docs):
            return "\n\n---\n\n".join(
                f"[来源 {i+1}] {doc.metadata.get('source', '未知')}\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )

        format_docs = RunnableLambda(_format_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个严格基于上下文回答问题的AI助手。请确保你的回答是基于提供的上下文。"),
            ("human", "上下文：\n{context}\n\n问题：{question}"),
        ])

        self.chain = (
            {"context": self.retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        """执行问答，返回答案和参考来源"""
        if not self._initialized:
            return {"success": False, "error": "服务未初始化，请先调用 initialize()"}
        try:
            answer = self.chain.invoke(question)
            docs = self.retriever.invoke(question)
            sources = [
                {"index": i + 1, "source": doc.metadata.get("source", "未知"),
                 "content_preview": doc.page_content[:100]}
                for i, doc in enumerate(docs)
            ]
            return {"success": True, "answer": answer, "sources": sources}
        except Exception as e:
            logger.exception("QAService ask 出错")
            return {"success": False, "error": str(e)}

    def reload(self) -> dict:
        """重新构建向量库"""
        try:
            docs = self._load_documents()
            QA_DB_DIR.mkdir(parents=True, exist_ok=True)
            # 清空旧库
            import shutil
            for f in QA_DB_DIR.iterdir():
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
            self.vectorstore = self._load_or_create_vectorstore(docs)
            self._setup_qa_chain()
            return {"success": True, "message": "向量库已重新构建"}
        except Exception as e:
            logger.exception("QAService reload 出错")
            return {"success": False, "error": str(e)}


# 全局单例
qa_service = QAService()
