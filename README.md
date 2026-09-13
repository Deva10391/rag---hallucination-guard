# Production RAG with Self-Evaluating Hallucination Guard

A retrieval-augmented generation system where a second, independent LLM pass verifies every answer's groundedness before it reaches the user — blocking hallucinated output rather than shipping it.

## Problem

Standard RAG assumes retrieval equals correctness. It doesn't check whether the *generated* answer actually stayed faithful to the retrieved text. Hallucination still slips through, and it's the single biggest blocker preventing companies from shipping LLM features in production.

## Solution

Retrieval and generation happen as normal — but every generated answer is passed to a second LLM "judge" that scores its faithfulness against the same retrieved chunks. Low-confidence answers are flagged or rejected instead of shown. The pipeline is benchmarked end-to-end with a standardized evaluation library, not a homemade metric.

## Speciality

- Typical RAG portfolio projects stop at retrieval + generation. This adds the verification layer that production RAG teams actually build.
- Every accepted answer is traceable to its supporting source chunk.
- Produces a hard, standardized metric: faithfulness/hallucination rate before vs. after the guard.

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

## Utilities Used

| Utility | Role |
|---|---|
| FAISS | Vector index and similarity search |
| sentence-transformers | Local, free embedding model |
| Groq API | Both the answer-generator and the independent judge LLM |
| FastAPI | Serving layer — gives a live, demoable endpoint |
| Ragas | Standardized RAG evaluation (faithfulness, groundedness) for the benchmark metric |

## Non-Functional Requirements

- **Groundedness threshold:** any answer below the confidence cutoff is rejected, never shown silently
- **Latency budget:** guard adds only one extra LLM call; total response time stays demo-viable
- **Source traceability:** every accepted answer cites its supporting chunk
- **Deployment persistence:** runs as a served endpoint, not just a notebook

## Simplified Working

It looks up the relevant paragraph, writes an answer, then — before showing you — a second reviewer double-checks whether the answer actually matches what the paragraph says or made something up. If it made something up, it's blocked. Like a fact-checker sitting between the writer and the reader.

## Metrics Reported

- Hallucination / faithfulness rate before vs. after guard (via Ragas)
- Rejection rate (how often the guard fires)
- Response latency with guard included

<hr>

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