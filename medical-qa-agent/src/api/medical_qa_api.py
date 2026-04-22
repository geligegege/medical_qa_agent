"""FastAPI 服务入口，提供 RAG 图工作流接口。"""

import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from starlette.responses import FileResponse

from src.graph.medical_qa_workflow import create_workflow
from src.graph.faiss_retriever_utils import load_faiss_index

warnings.filterwarnings("ignore")


class Question(BaseModel):
    question: str


api_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 生命周期事件的异步上下文管理器。"""
    try:
        # 启动时加载向量索引
        faiss_index = load_faiss_index()
        # 基于索引创建工作流并缓存到上下文
        logger.info("Creating the workflow...")
        api_context["workflow"] = create_workflow(faiss_index)
        yield
    except Exception:
        logger.exception("Failed to load FAISS index and create the workflow.")
        raise HTTPException(
            status_code=500,
            detail="Failed to load FAISS index and create the workflow.",
        )
    del faiss_index
    del api_context["workflow"]
    logger.info("Workflow deleted.")


app = FastAPI(title="Rag Graph API", version="0.1.0", lifespan=lifespan)


static_path = os.path.join(os.path.dirname(__file__), "static")
print(static_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
def read_root():
    return FileResponse(static_path + "/index.html")


@app.post("/answer")
async def answer(question: Question):
    """接收用户问题并返回工作流推理结果。"""
    try:
        # 调用图工作流执行检索与生成
        graph = api_context["workflow"]
        state = graph.invoke({"question": question.question})
        logger.info(f"Response: {state}")
        return JSONResponse(content=state)
    except Exception:
        logger.exception("Failed to answer the question.")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question.",
        )


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
