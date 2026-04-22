from typing import Any, Dict

import torch
import torch._inductor.config
from llm_guard import scan_output
from llm_guard.output_scanners import LanguageSame, Relevance

from src.config import settings
from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True

language_same_scanner = LanguageSame(use_onnx=True)
relevance_scanner = Relevance(use_onnx=True)


def check_language_same(state: AgentState) -> Dict[str, Any]:
    """执行回答与提问语言一致性检查。"""
    output = state["llm_output"]
    prompt = state["prompt"]
    _, results_valid, _ = scan_output(
        scanners=[language_same_scanner], output=output, prompt=prompt
    )
    same_language = not results_valid.get("LanguageSame", True)
    return {"answer_status": [1 if same_language else 0]}


def check_relevance(state: AgentState) -> Dict[str, Any]:
    """执行回答与提问相关性检查。"""
    output = state["llm_output"]
    prompt = state["prompt"]
    _, results_valid, _ = scan_output(
        scanners=[relevance_scanner], output=output, prompt=prompt
    )
    relevant_answer = not results_valid.get("Relevance", True)
    return {"answer_status": [1 if relevant_answer else 0]}


def answer_check_node(state: AgentState) -> Dict[str, Any]:
    """汇总回答检查结果并返回最终状态。"""
    answer_status = state["answer_status"]
    answer = state["llm_output"]
    rewrite_count = state.get("rewrite_count", 0)
    max_retries = settings.ANSWER_REWRITE_MAX_RETRIES
    all_checks_passed = all(status == 0 for status in answer_status[-2:])

    if all_checks_passed:
        state["answer_valid"] = True
        return {
            "llm_output": answer,
            "answer_valid": True,
            "answer_route": "end",
            "rewrite_hint": "",
        }

    if rewrite_count < max_retries:
        # 未超过最大重写次数时，回到生成节点进行重写
        next_count = rewrite_count + 1
        return {
            "answer_valid": False,
            "answer_route": "retry",
            "rewrite_count": next_count,
            "rewrite_hint": "上一版回答未通过一致性或相关性检查，请仅基于检索上下文重写并提高相关性与语言一致性。",
        }

    state["answer_valid"] = False
    return {"llm_output": "Answer failed checks, please try again."}


if __name__ == "__main__":
    state = {"llm_output": "Hello, how can I assist you today?", "prompt": "Hello"}
    check_language_same(state)

    state2 = {
        "llm_output": "Bonjour, comment puis-je vous aider aujourd'hui?",
        "prompt": "Hello",
    }
    check_language_same(state2)

    # 使用不相关示例测试相关性检查
    state3 = {
        "llm_output": "Hello, how can I assist you today?",
        "prompt": "What is the weather today?",
    }
    check_relevance(state3)

    # 使用相关示例测试相关性检查
    state4 = {
        "llm_output": "The weather today is sunny.",
        "prompt": "What is the weather today?",
    }
    check_relevance(state4)
