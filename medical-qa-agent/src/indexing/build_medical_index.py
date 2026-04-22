"""预处理数据集并构建 FAISS 索引。"""

import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import polars as pl
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# 本地模块导入
from src.config import settings


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _ensure_local_dataset_path() -> Path:
    """确保数据集位于本地并返回本地路径。"""
    raw_data = str(settings.RAW_DATA_PATH)
    data_url = str(settings.DATA_URL)

    if not _is_url(raw_data):
        local_path = Path(raw_data)
        if local_path.is_file():
            return local_path
        source_url = data_url
    else:
        local_path = Path(settings.DATA_DIR) / "medical_data.jsonl"
        source_url = raw_data

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {source_url} to {local_path} ...")
    urlretrieve(source_url, local_path)
    logger.info("Dataset download complete.")
    return local_path


# 使用 Polars 加载并清洗数据集
def download_and_preprocess_dataset() -> pl.DataFrame:
    """下载并预处理数据集。"""
    # 从本地文件读取，避免 Polars 直接远程读取时出现重试抖动
    local_dataset_path = _ensure_local_dataset_path()
    logger.info(f"Reading dataset from {local_dataset_path} ...")
    customer_care_df = pl.read_ndjson(local_dataset_path)
    logger.info(f"Loaded dataset with {customer_care_df.height} records.")

    # 统一字段名并清理空值
    customer_care_df = customer_care_df.select(["Question", "Must_have"]).rename(
        {"Question": "question", "Must_have": "answer"}
    )
    customer_care_df = customer_care_df.drop_nulls()
    logger.info(f"Preprocessed dataset with {customer_care_df.height} records.")

    return customer_care_df


def generate_documents(customer_care_df: pl.DataFrame) -> list[Document]:
    """将 DataFrame 记录转换为 LangChain 文档对象。"""
    documents = [
        Document(
            page_content=row["question"],
            metadata=row,
            id=idx,
        )
        for idx, row in enumerate(customer_care_df.to_dicts())
    ]
    logger.info(f"Generated {len(documents)} documents.")
    return documents


def faiss_index_exists(index_path: str) -> bool:
    """判断给定路径下的 FAISS 索引是否完整可用。"""
    path = Path(index_path)
    return (path / "index.faiss").is_file() and (path / "index.pkl").is_file()


def create_faiss_index(documents: list[Document]) -> None:
    """创建或更新 FAISS 索引，并避免重复入库。"""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    index_path = settings.FAISS_INDEX_PATH

    if faiss_index_exists(index_path):
        # 加载已有索引
        logger.info("Loading existing FAISS index...")
        faiss_index = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        # 收集已有文档 ID
        existing_ids = set(faiss_index.index_to_docstore_id.values())
        # 仅保留新增文档
        new_docs = [doc for doc in documents if doc.id not in existing_ids]
        if new_docs:
            logger.info(f"Adding {len(new_docs)} new documents.")
            faiss_index.add_documents(new_docs)
            faiss_index.save_local(index_path)
            logger.info(f"Updated index saved to {index_path}")
        else:
            logger.info("No new documents to add.")
    else:
        # 首次构建新索引
        if os.path.exists(index_path):
            logger.warning(
                f"FAISS index path exists but is incomplete; rebuilding at {index_path}"
            )
        logger.info("Creating new FAISS index...")
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(index_path)
        logger.info(f"New index saved to {index_path}")


def embed_and_index():
    """执行向量化与索引构建流程。"""
    # 下载并预处理数据
    customer_care_df = download_and_preprocess_dataset()

    # 生成文档对象
    documents = generate_documents(customer_care_df)

    # 创建或更新 FAISS 索引
    create_faiss_index(documents)


if __name__ == "__main__":
    embed_and_index()
