"""助手的图工作流定义。"""

from functools import partial

from langchain_core.globals import set_debug
from langgraph.graph import END, START, StateGraph

# 本地模块导入
from src.graph.answer_guard_node import (
    answer_check_node,
    check_language_same,
    check_relevance,
)
from src.graph.answer_generation_node import answer_node
from src.graph.context_grading_node import grade_documents_node
from src.graph.input_guard_node import (
    question_check_node,
    scan_prompt_injection,
    scan_token_limit,
    scan_toxicity,
)
from src.graph.context_retrieval_node import retrieve
from src.graph.workflow_state import AgentState
from src.graph.medical_topic_gate_node import topic_classifier
from src.graph.faiss_retriever_utils import load_faiss_index

set_debug(True)


def create_workflow(retriever):
    """构建并编译工作流。"""
    workflow = StateGraph(AgentState)
    workflow.add_node(
        "scan_prompt_injection",
        scan_prompt_injection,
    )
    workflow.add_node(
        "scan_toxicity",
        scan_toxicity,
    )
    workflow.add_node(
        "scan_token_limit",
        scan_token_limit,
    )
    workflow.add_node("question_check_node", question_check_node)
    workflow.add_conditional_edges(
        "question_check_node",
        lambda state: state["question_valid"],
        {True: "topic_classifier", False: END},
    )
    workflow.add_node("topic_classifier", topic_classifier)
    workflow.add_conditional_edges(
        "topic_classifier",
        lambda state: state["on_topic"],
        {
            "Yes": "retrieve_docs",
            "No": END,
        },
    )
    workflow.add_node("retrieve_docs", partial(retrieve, faiss_retriever=retriever))
    workflow.add_node("docs_grader", grade_documents_node)
    workflow.add_node("check_language_same", check_language_same)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("answer_check_node", answer_check_node)

    workflow.add_node("generate_answer", answer_node)

    workflow.add_edge(START, "scan_prompt_injection")
    workflow.add_edge(START, "scan_toxicity")
    workflow.add_edge(START, "scan_token_limit")
    workflow.add_edge("scan_prompt_injection", "question_check_node")
    workflow.add_edge("scan_toxicity", "question_check_node")
    workflow.add_edge("scan_token_limit", "question_check_node")
    workflow.add_edge("retrieve_docs", "docs_grader")
    workflow.add_edge("docs_grader", "generate_answer")
    workflow.add_edge("generate_answer", "check_language_same")
    workflow.add_edge("generate_answer", "check_relevance")
    workflow.add_edge("check_language_same", "answer_check_node")
    workflow.add_edge("check_relevance", "answer_check_node")
    workflow.add_conditional_edges(
        "answer_check_node",
        lambda state: state.get("answer_route", "end"),
        {
            "retry": "generate_answer",
            "end": END,
        },
    )

    graph = workflow.compile()
    return graph


if __name__ == "__main__":
    # 加载 FAISS 索引
    faiss_retriever = load_faiss_index()

    app = create_workflow(faiss_retriever)
    app.get_graph().draw_mermaid_png(output_file_path="/assets/flow.png")

    # 运行工作流

    config = {"configurable": {"thread_id": 1}}

    state1 = {"question": "What is the capital of France?"}
    state2 = {"question": "I wnat to return a package"}
    final_state1 = app.invoke(state1, config=config)
    final_state2 = app.invoke(state2, config=config)
