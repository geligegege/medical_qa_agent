"""
RAG 系统评估器。

该模块通过文档采样、答案生成和指标计算，使用 RAGAS 对 RAG 系统进行评估。
"""

import os
import random
from uuid import uuid4

from langchain_openai import ChatOpenAI
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity
from loguru import logger
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.config import settings
from src.graph.medical_qa_workflow import create_workflow
from src.graph.faiss_retriever_utils import load_faiss_index


def setup_components():
    """初始化 RAG 评估所需的全部组件。"""
    input_scanners = [PromptInjection(), TokenLimit(), Toxicity()]
    retriever = load_faiss_index()
    rag_app = create_workflow(retriever, input_scanners=input_scanners)

    llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0, max_tokens=1000)
    evaluator_llm = LangchainLLMWrapper(llm)

    return retriever, rag_app, evaluator_llm


def prepare_evaluation_data(retriever, rag_app):
    """采样文档并构建评估数据集。"""
    # 按固定随机种子进行采样，确保评估结果可复现
    sample_size = settings.EVALUATION_SAMPLE_SIZE
    random.seed(settings.EVALUATION_RANDOM_SEED)
    # 直接从向量库底层文档存储中提取原始文档
    documents = list(retriever.vectorstore.docstore._dict.values())
    sampled_docs = random.sample(documents, min(sample_size, len(documents)))

    logger.info(f"正在处理 {len(sampled_docs)} 条文档...")

    # 组装 RAGAS 所需的数据结构
    dataset = []
    for i, doc in enumerate(sampled_docs, 1):
        try:
            query = doc.metadata.get("question", "")
            reference = doc.metadata.get("answer", "")
            # 每次调用使用独立线程 ID，避免图状态互相污染
            thread_id = str(uuid4())

            # 先检索上下文，再调用工作流生成回答
            retrieved_docs = retriever.get_relevant_documents(query)
            retrieved_contexts = [d.metadata for d in retrieved_docs]

            # 将检索结果标准化为字符串列表，便于评估器消费
            cleaned_contexts = [
                f"question: {d['question']}\nanswer: {d['answer']}"
                for d in retrieved_contexts
            ]

            config = {"configurable": {"thread_id": thread_id}}
            state = {"question": query}
            response = rag_app.invoke(state, config=config)
            response = response.get("llm_output", "")

            dataset.append(
                {
                    "user_input": query,
                    "retrieved_contexts": cleaned_contexts,
                    "response": response,
                    "reference": reference,
                }
            )

            logger.info(f"已处理 {i}/{len(sampled_docs)}")

        except Exception as e:
            logger.info(f"处理第 {i} 条文档时发生错误: {e}")
            continue

    return dataset


def run_evaluation(dataset, evaluator_llm):
    """执行 RAGAS 评估并保存结果。"""
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

    logger.info("正在运行 RAGAS 评估...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # 转换为 DataFrame 便于导出与查看
    output_dir = settings.EVALUATION_OUTPUT_DIR
    results_df = results.to_pandas()
    # 保存明细结果与均值结果
    results_html_path = os.path.join(output_dir, "evaluation_results.html")
    results_df.to_html(results_html_path, index=False)
    mean_scores = results_df.mean(numeric_only=True).round(4).to_frame(name="score")
    mean_scores_path = os.path.join(output_dir, "mean_scores.html")
    mean_scores.to_html(mean_scores_path)
    logger.info(f"评估结果已保存到 {output_dir}")


def main():
    """评估主流程入口。"""
    logger.info("开始执行 RAG 评估...")

    # 初始化依赖组件
    retriever, rag_app, evaluator_llm = setup_components()

    # 准备评估数据并执行评估
    dataset = prepare_evaluation_data(retriever, rag_app)
    run_evaluation(dataset, evaluator_llm)

    logger.info("评估完成！")


if __name__ == "__main__":
    main()
