from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.llm_client_factory import create_chat_llm
from src.graph.workflow_state import AgentState


class GradeDocuments(BaseModel):
    """用于检索文档相关性判断的二分类结构。"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieval_grader(doc: str, question: str, local_llm: Optional[bool] = None):
    """创建并运行文档相关性评分器。"""

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )
    use_local_llm = settings.USE_LOCAL_LLM if local_llm is None else local_llm
    llm = create_chat_llm(use_local_llm)
    if use_local_llm:
        # 本地模型直接返回文本，后续按 yes/no 解析
        retrieve_grader = grade_prompt | llm
        grader_output = retrieve_grader.invoke({"question": question, "document": doc})
        return grader_output.content

    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieve_grader = grade_prompt | structured_llm_grader
    grader_output = retrieve_grader.invoke({"question": question, "document": doc})
    return grader_output.binary_score


def grade_documents_node(state: AgentState):
    """过滤检索结果，仅保留与问题相关的文档。"""
    docs = state["documents"]
    question = state["question"]
    filtered_docs = []
    for doc in docs:
        grade = retrieval_grader(doc, question, local_llm=None)
        if grade.lower() == "yes":
            filtered_docs.append(doc)
    return {"documents": filtered_docs}
