# evaluation/ragas_eval.py

import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from golden_dataset import GOLDEN_DATASET



load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/query"


def collect_pipeline_outputs() -> list[dict]:
    """
    Run every golden question through the live RAG API and collect
    question, generated answer, retrieved contexts, and ground truth.
    """
    records = []

    for example in GOLDEN_DATASET:
        payload = {
            "question":    example.question,
            "ticker":      example.ticker,
            "form_type":   example.form_type,
            "fiscal_year": example.fiscal_year,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            records.append({
                "question":     example.question,
                "answer":       data["answer"],
                "contexts":     [s["text"] for s in data["sources"]],
                "ground_truth": example.ground_truth,
            })
            logger.info(f"✓ {example.question[:60]}")

        except Exception as e:
            logger.error(f"✗ Failed on '{example.question[:60]}': {e}")
            records.append({
                "question":     example.question,
                "answer":       "",
                "contexts":     [],
                "ground_truth": example.ground_truth,
            })

    return records


def run_evaluation():
    logger.info(f"Running evaluation on {len(GOLDEN_DATASET)} golden examples...")
    records = collect_pipeline_outputs()

    # Build RAGAS 0.2.x EvaluationDataset from SingleTurnSamples
    samples = []
    for r in records:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        ))
    dataset = EvaluationDataset(samples=samples)

    # Wrap Groq + HuggingFace in RAGAS wrappers
    judge_llm = LangchainLLMWrapper(ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Instantiate metrics with the judge LLM/embeddings
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    logger.info("Scoring with RAGAS...")

    run_config = RunConfig(
        max_retries=5,
        max_wait=60,
        timeout=120,
    )

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )

    df = results.to_pandas()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("evaluation/results", exist_ok=True)
    detail_path = f"evaluation/results/eval_{timestamp}.csv"
    df.to_csv(detail_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric in metric_cols:
        if metric in df.columns:
            avg = df[metric].mean()
            print(f"  {metric:22s}: {avg:.3f}")
    print("=" * 60)
    print(f"\nDetailed results → {detail_path}")

    if "faithfulness" in df.columns:
        print("\nLowest faithfulness scores (your debugging targets):")
        print(df.columns.tolist())   # temporary — shows you the real column names
        worst = df.nsmallest(3, "faithfulness")
        for _, row in worst.iterrows():
            question = row.get("user_input", row.get("question", "N/A"))
            score = row["faithfulness"] if not pd.isna(row["faithfulness"]) else 0.0
            print(f"  {score:.2f}  {str(question)[:70]}")

    return df


if __name__ == "__main__":
    run_evaluation()