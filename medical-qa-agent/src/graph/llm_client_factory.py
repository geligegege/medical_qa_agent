"""大语言模型工厂辅助函数。"""

from typing import Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import settings


def create_chat_llm(local_llm: Optional[bool] = None):
    """创建本地 Ollama 或云端千问兼容 LLM 实例。"""
    use_local_llm = settings.USE_LOCAL_LLM if local_llm is None else local_llm

    if use_local_llm:
        # 本地推理路径
        return ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    api_key = (
        settings.DASHSCOPE_API_KEY.get_secret_value()
        if settings.DASHSCOPE_API_KEY
        else ""
    )
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is required when USE_LOCAL_LLM=false.")

    # 云端推理路径
    return ChatOpenAI(
        model=settings.QWEN_MODEL_NAME,
        api_key=api_key,
        base_url=settings.QWEN_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
