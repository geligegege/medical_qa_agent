from operator import add
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    """图工作流共享状态定义。"""

    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    on_topic: bool
    prompt: str
    llm_output: str
    documents: List[str]
    answer_status: Annotated[list, add]
    answer_valid: bool
    rewrite_count: int
    rewrite_hint: str
    answer_route: str
