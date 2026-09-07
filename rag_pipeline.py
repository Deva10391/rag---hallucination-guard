import os
import json
import re
import faiss
import config
import threading

import numpy as np

from sentence_transformers import SentenceTransformer
from groq import AsyncGroq

_embed_model = None
_index = None
_chunks = None
_groq_client = None
_load_lock = threading.Lock()

def gen_context(retrieved: list[dict]) -> str:
    res = "\n\n".join(
        f"[{i+1}] (source: {r['source']}) {r['text']}" for i, r in enumerate(retrieved)
    )
    return res

async def llm_gen(prompt: str, model: str) -> str:
    response = await _groq_client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def _lazy_load():
    global _embed_model, _index, _chunks, _groq_client
    if _embed_model is not None and _index is not None and _groq_client is not None:
        return
    with _load_lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer(config.EMBED_MODEL)
        if _index is None:
            index_path = os.path.join(config.INDEX_DIR, "faiss.index")
            chunk_path = os.path.join(config.INDEX_DIR, "chunks.json")
            if not os.path.exists(index_path):
                raise RuntimeError(
                    "No index found\nyou will need to run `python ingest.py` first, it generates the FAISS index :)"
                )
            _index = faiss.read_index(index_path)
            with open(chunk_path, "r", encoding="utf-8") as f:
                _chunks = json.load(f)
        if _groq_client is None:
            if not config.GROQ_API_KEY:
                raise RuntimeError(
                    "pleas add GROQ_API_KEY to that .env file of yours"
                )
            _groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)

# ofc, to get those vectors
def retrieve(question: str, top_k: int = None) -> list[dict]:
    _lazy_load()
    top_k = top_k or config.TOP_K
    q_emb = _embed_model.encode([question], normalize_embeddings=True).astype("float32") # as it was saved such
    scores, idxs = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        record = dict(_chunks[idx])
        record["similarity"] = float(score)
        results.append(record)
    return results


async def generate_answer(question: str, retrieved: list[dict]) -> str:
    _lazy_load()
    context = gen_context(retrieved)
    prompt = (
        f"""given the context, and only the context, answer the question asked; if you evaluate that context DOES NOT contain the answer, say 'I don't know' explicitly and without citation marker(s); otherwise, do mark the bracket number(s) of the chunks used in formar [1], [2], ... [i] format

        CONTEXT:
        {context}

        QUESTION:
        {question}

        lets go, please do answer :)
        """
    )
    response = await llm_gen(prompt, config.GEN_MODEL)
    return response

async def judge_groundedness(question: str, answer: str, retrieved: list[dict]) -> dict:
    _lazy_load()
    context = gen_context(retrieved)
    prompt = (
        f"""given the context and the answer - provided as they're - you're requested to STRICTLY fact-check whether or not EACH CLAIM of the answer, is DIRECTLY SUPPORTED by the context. here, please discard the outside-of-context knowledge"

        CONTEXT:
        {context}
        
        ANSWER:
        {answer}

        """
        "also, for the response, use the strict-json format of-\n"
        '{"score": <float (between 0,1 - where 1=fully grounded, 0=fabricated)>, "reasoning": "<one sentence only >"}'
    )
    raw = await llm_gen(prompt, model = config.JUDGE_MODEL)
    try:
        cleaned = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
            score = 0.0
            reasoning = f"Can not parse the judge model's response: {raw[:100]}"
    return {"score": score, "reasoning": reasoning}

def extract_cited_sources(answer: str, retrieved: list[dict]) -> list[dict]:
    cited_idxs = {int(n) for n in re.findall(r"[\[【](\d+)[\]】]", answer)}
    cited = []
    for i, r in enumerate(retrieved, start=1):
        if i in cited_idxs:
            cited.append({
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "similarity": r["similarity"],
                "text": r["text"],
            })
    return cited

# this is the one, THE one, the rubber-wonder to do it all
async def answer_with_guard(question: str, top_k: int = None, threshold: float = None) -> dict:
    threshold = threshold if threshold is not None else config.FAITHFULNESS_THRESHOLD
    retrieved = retrieve(question, top_k)

    if not retrieved:
        return {
            "status": "rejected",
            "reason": "no relevant context found",
            "answer": None,
            "sources": [],
            "all_retrieved": [],
            "faithfulness_score": 0.0,
        }

    answer = await generate_answer(question, retrieved)
    verdict = await judge_groundedness(question, answer, retrieved)
    cited_sources = extract_cited_sources(answer, retrieved)

    is_refusal = answer.strip().lower().startswith("i don't know") or answer.strip().lower().startswith("i do not know")
    accepted = verdict["score"] >= threshold and (is_refusal or bool(cited_sources))

    if verdict["score"] >= threshold and not accepted:
        reason = f"faithfulness passed ({verdict["score"]}) but no citation marker found in answer - rejected"
    elif verdict["score"] < threshold:
        reason = f"below faithfulness threshold ({threshold}): {verdict["reasoning"]}"
    else:
        reason = None

    cited_sources = cited_sources if accepted else []
    
    return {
        "status": "accepted" if accepted else "rejected",
        "reason": reason,
        "answer": answer if accepted else None,
        "raw_answer": answer,
        "sources": cited_sources,
        "all_retrieved": [{"source": r["source"], "similarity": r["similarity"]} for r in retrieved],
        "faithfulness_score": verdict["score"],
        "judge_reasoning": verdict["reasoning"],
    }