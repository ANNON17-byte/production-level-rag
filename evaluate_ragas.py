"""STEP 6 - Evaluate the pipeline with RAGAS, using your local LM Studio model as the judge.

Metrics:
  context_precision : are the useful chunks ranked near the top of what was retrieved?
  context_recall    : did retrieval fetch everything needed to produce the reference answer?
  faithfulness      : is the answer supported by the retrieved chunks (no hallucination)?

Fill eval_set.json with 10-30 question/reference pairs first.
"""
import json

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.run_config import RunConfig

from config import BASE_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from generation import generate_answer
from retrieval import FaissRetriever


def main():
    retriever = FaissRetriever()
    eval_items = json.loads((BASE_DIR / "eval_set.json").read_text(encoding="utf-8"))

    # 1) Run YOUR pipeline on every question and record what it did
    rows = []
    for item in eval_items:
        hits = retriever.retrieve(item["question"])
        rows.append({
            "user_input": item["question"],                       # RAGAS field names
            "retrieved_contexts": [h["text"] for h in hits],
            "response": generate_answer(item["question"], hits),
            "reference": item["reference"],
        })
        print("done:", item["question"][:60])

    dataset = EvaluationDataset.from_list(rows)

    # 2) The "judge": RAGAS asks this LLM many small yes/no style questions
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0)
    )

    # 3) Score. max_workers=1 because one local GPU can only handle one request at a time
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness()],
        llm=judge,
        run_config=RunConfig(timeout=600, max_workers=1),
    )

    print(result)                       # average score per metric
    df = result.to_pandas()
    df.to_csv(BASE_DIR / "ragas_results.csv", index=False)
    print("Per-question scores saved to ragas_results.csv")


if __name__ == "__main__":
    main()
