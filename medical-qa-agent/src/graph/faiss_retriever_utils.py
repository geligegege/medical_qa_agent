from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import settings


def load_faiss_index() -> FAISS:
    """加载 FAISS 索引并返回检索器对象。"""
    try:
        logger.info("Loading FAISS index...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        # 记录原始异常并继续向上抛出，便于上层统一处理
        logger.exception("Failed to load FAISS index.")
        raise e

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )

    return retriever
