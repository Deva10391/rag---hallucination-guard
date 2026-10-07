# Production RAG with Self-Evaluating Hallucination Guard
 
## Problem
 
RAG assumes retrieved context makes the answer correct. The generated answer can still drift from the retrieved text.
 
## Solution
 
Retrieve top-4 chunks (FAISS inner product over normalized sentence-transformers embeddings, 500-word chunks with 75-word overlap). A Groq LLM answers using only those chunks with `[n]` citation markers. A second Groq LLM scores the answer's faithfulness 0–1 against the same chunks. The answer is shown only if the score is ≥ 0.7 (configurable) and it carries a citation marker, or is an explicit "I don't know". Otherwise the API returns `rejected` with the reason.
 
## Speciality
 
- Two guard rules: faithfulness threshold + mandatory citation marker.
- Accepted answers return their cited source chunks.
- Ragas faithfulness is run offline over the pipeline (`eval_ragas.py`).
- Measured result (`results.txt`, 4 questions): faithfulness 0.83 on answers shown, 25% of questions rejected by the guard. Raw and guarded faithfulness are both 0.83 on this small set.
## Architecture
 
```
Query
  │
  ▼
┌──────────────┐     ┌───────┐      ┌─────────────┐
│ sentence-    │ ──▶ │ FAISS │ ──▶ │ Top-k chunks│
│ transformers │     │ index │      └─────────────┘
└──────────────┘     └───────┘             │
                                           ▼
                                 ┌────────────────────┐
                                 │ Groq LLM #1        │
                                 │ (answer generator) │
                                 └────────────────────┘
                                           │
                                           ▼
                                 ┌─────────────────────┐
                                 │ Groq LLM #2         │
                                 │ (faithfulness judge)│
                                 └─────────────────────┘
                                           │
                          pass ────────────┼──────────── fail
                            │                              │
                            ▼                              ▼
                    Answer + citation             Flagged / rejected
 
              (Ragas runs offline over the full pipeline to produce benchmark metrics)
```
 
## Simplified Working
 
It finds the relevant paragraphs, writes a cited answer, and a second model checks that answer against those paragraphs before you see it. Unsupported answers are blocked.
 
## Utilities
 
- FAISS (`IndexFlatIP`) — vector search
- sentence-transformers — local embeddings
- Groq API — generator and independent judge
- FastAPI — `POST /query`, `GET /health`
- Ragas — faithfulness benchmark
## To Run
 
for installation,
```
pip install -r requirements.txt
python ingest.py
```
to run, in cli, run-
```
uvicorn app:app --reload
```
 
then, in another cli
```
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is the hallucination guard threshold?\"}"
```
 
## To Get Metrics/ Performance
 
run-
 
```
python eval_ragas.py
```