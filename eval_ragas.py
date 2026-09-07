import json
import config
import asyncio

from ragas.metrics.collections import Faithfulness
from ragas.llms import llm_factory
from openai import AsyncOpenAI

from rag_pipeline import answer_with_guard

TEST_SET = [
    "what does hallucination guard EVEN check huh?",
    "you say you check before returning an answer - how?",
    "whats your stored vector store?",
    "whatcha do if faithfulness falls below threshold?",
]

REFUSAL_PHRASES = ["i don't know", "i do not know", "cannot answer", "no information"]

client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
judge_llm = llm_factory(config.JUDGE_MODEL, client=client)
faithfulness = Faithfulness(llm=judge_llm)

async def score_one(question, answer, contexts):
    result = await faithfulness.ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    return result.value

async def run_eval():
    raw_scores = []
    guarded_scores = []
    skipped = 0

    for question in TEST_SET:
        print(f"asking question: {question}")
        result = await answer_with_guard(question)
        answer = (result.get("raw_answer") or "").strip()
        contexts = [s["text"] for s in result["sources"] if s.get("text")]

        is_refusal = any(p in answer.lower() for p in REFUSAL_PHRASES)

        print(question, result["status"], result.get("reason"))
        if is_refusal or not contexts:
            skipped += 1
            continue
        
        score = await score_one(question, answer, contexts)
        raw_scores.append(score)
        if result["status"] == "accepted":
            guarded_scores.append(score)
        print(f"{guarded_scores}")

    def avg(scores):
        return sum(scores) / len(scores) if scores else None

    print(json.dumps({
        "n_questions": len(TEST_SET),
        "n_scored_raw": len(raw_scores),
        "n_passed_guard": len(guarded_scores),
        "n_skipped_refusals_or_no_context": skipped,
        "raw_faithfulness_all_answers": avg(raw_scores),
        "guarded_faithfulness_shown_to_user": avg(guarded_scores),
        "rejection_score": 1 - (len(guarded_scores) / len(TEST_SET)) if TEST_SET else 0
    }, indent=2))

if __name__ == "__main__":
    asyncio.run(run_eval())