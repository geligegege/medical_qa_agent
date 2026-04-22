"""项目配置定义。"""

import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """项目设置项。"""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # 基础路径
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # 数据配置
    DATA_URL: str = (
        "https://huggingface.co/datasets/Itaykhealth/K-QA/resolve/main/questions_w_answers.jsonl"
    )
    RAW_DATA_PATH: str = str(DATA_DIR / "medical_data.jsonl")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.csv")

    # 向量嵌入模型配置
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 通用 LLM 配置
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 100

    # 本地 LLM 配置
    USE_LOCAL_LLM: bool = False
    OLLAMA_MODEL_NAME: str = "llama3.2:3b"

    # 千问云端 LLM 配置
    DASHSCOPE_API_KEY: Union[SecretStr, None] = None
    QWEN_MODEL_NAME: str = "qwen-plus"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")

    FAISS_TOP_K: int = 5

    # OpenAI 接口配置
    OPENAI_API_KEY: Union[SecretStr, None] = None

    # 评估配置
    EVALUATION_SAMPLE_SIZE: int = 10
    EVALUATION_OUTPUT_DIR: str = str(BASE_DIR / "evaluation_results")
    EVALUATION_RANDOM_SEED: int = 123

    # 回答重写配置
    ANSWER_REWRITE_MAX_RETRIES: int = 2

    # 日志配置
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # 初始化时确保必要目录存在
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)
        os.makedirs(self.EVALUATION_OUTPUT_DIR, exist_ok=True)


settings = Settings()
