from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.llm_client_factory import create_chat_llm
from src.graph.workflow_state import AgentState


def generate_answer(
    question: str,
    context: list,
    rewrite_hint: str = "",
    previous_answer: str = "",
    local_llm: Optional[bool] = None,
):
    """基于检索到的上下文生成问题回答。"""
    llm = create_chat_llm(local_llm)
    # 通过模板约束模型仅依据给定上下文回答；若存在重写提示则引导重写
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Rewrite hint (empty means first attempt): {rewrite_hint}
    Previous answer (may be empty): {previous_answer}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    formatted_prompt = prompt.format(
        question=question,
        context=context,
        rewrite_hint=rewrite_hint,
        previous_answer=previous_answer,
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {
            "question": question,
            "context": context,
            "rewrite_hint": rewrite_hint,
            "previous_answer": previous_answer,
        }
    )
    return result, formatted_prompt


def answer_node(state: AgentState):
    """回答生成节点。"""
    question = state["question"]
    context = state["documents"]
    rewrite_hint = state.get("rewrite_hint", "")
    previous_answer = state.get("llm_output", "")
    answer, prompt = generate_answer(
        question,
        context,
        rewrite_hint=rewrite_hint,
        previous_answer=previous_answer,
        local_llm=None,
    )
    return {"llm_output": answer, "prompt": prompt}
