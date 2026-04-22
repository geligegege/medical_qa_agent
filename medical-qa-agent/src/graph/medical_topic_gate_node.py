from functools import lru_cache
from typing import Any, Dict, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.llm_client_factory import create_chat_llm
from src.graph.workflow_state import AgentState


# 主题分类的结构化输出定义
class GradeTopic(BaseModel):
    """主题分类结果结构。"""

    score: Literal["Yes", "No"] = Field(
        description="Whether the question is about customer support."
    )


@lru_cache(maxsize=100)
def classify_topic(question: str, local_llm: Optional[bool] = None) -> Dict[str, Any]:
    system = """You are a grader assessing whether a user's question is related to customer support 
    for a product or a purchase.
    Customer support topics include:
    - Questions about purchasing products (e.g., "How do I place an order?")
    - Questions about order cancellations (e.g., "Can I cancel my order?")
    - Questions about refunds or returns (e.g., "How do I request a refund?")
    - Questions about product issues (e.g., "My product is not working.")
    - Questions about account issues (e.g., "I can't log in to my account.")

    If the question is about customer support, respond with "Yes". Otherwise, respond with "No".
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )

    llm = create_chat_llm(local_llm)

    # 使用结构化输出可降低模型返回格式不稳定的问题
    structured_llm = llm.with_structured_output(GradeTopic)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    return result


def topic_classifier(state: AgentState):
    """判断问题是否属于客服主题。"""
    question = state["question"]
    result = classify_topic(question, local_llm=None)
    print(result)

    # 若后续需要，可在低置信度时设置保守兜底策略
    # 示例：state["on_topic"] = result.score
    if result.score == "Yes":
        return {"on_topic": "Yes"}
    else:
        return {
            "on_topic": "No",
            "llm_output": "Please ask a question about customer support so I can help you better.",
        }
