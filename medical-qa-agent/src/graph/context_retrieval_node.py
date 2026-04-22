from typing import Any, Dict

from src.graph.workflow_state import AgentState


def retrieve(state: AgentState, faiss_retriever) -> Dict[str, Any]:
    """从 FAISS 索引中检索与问题相关的文档。"""

    # 读取当前问题
    question = state["question"]

    # 执行向量检索
    documents = faiss_retriever.invoke(question)

    # 提取文档元数据作为后续节点输入
    metadata = [doc.metadata for doc in documents]

    # 返回增量状态，由图引擎合并到全局状态
    # 示例：也可直接写入 state["documents"] = metadata

    return {"documents": metadata}
