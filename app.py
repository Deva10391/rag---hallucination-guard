from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import answer_with_guard
import config

app = FastAPI(title="RAG + Hallucination Guard")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None

class QueryResponse(BaseModel):
    status: str
    answer: str | None
    faithfulness_score: float
    reason: str | None
    sources: list[dict]
    all_retrieved: list[dict]

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = await answer_with_guard(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        status=result["status"],
        answer=result["answer"],
        faithfulness_score=result["faithfulness_score"],
        reason=result["reason"],
        sources=result["sources"],
        all_retrieved=result["all_retrieved"],
    )