"""
FastAPI REST API Service for Twitter Customer Support Pipeline.
Endpoints:
- GET /health
- POST /predict
- GET /metrics
- GET /failures
"""

import json
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.orchestrator import SupportPipeline, PipelineOutput
from eval.runner import run_full_evaluation

app = FastAPI(
    title="Twitter Customer Support AI Pipeline (@SpotifyCares)",
    description="LLM-orchestrated support pipeline with grounded retrieval and hybrid escalation gating.",
    version="1.0.0"
)

# Lazy-loaded pipeline singleton
_pipeline: Optional[SupportPipeline] = None


def get_pipeline() -> SupportPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SupportPipeline()
    return _pipeline


class PredictRequest(BaseModel):
    message: str = Field(..., example="I was charged $10.99 twice for Spotify Premium!")
    turn_count: int = Field(default=1, ge=1, example=1)
    top_k: int = Field(default=3, ge=1, le=10, example=3)


@app.get("/health")
def health():
    return {"status": "ok", "service": "spotify-support-pipeline", "version": "1.0.0"}


@app.post("/predict", response_model=PipelineOutput)
def predict(req: PredictRequest):
    try:
        pipeline = get_pipeline()
        return pipeline.process(
            customer_message=req.message,
            turn_count=req.turn_count,
            top_k=req.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    eval_file = "eval/eval_results.json"
    if not os.path.exists(eval_file):
        run_full_evaluation(output_path=eval_file)

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "summary_table": data.get("summary_table", []),
        "human_agreement": data.get("human_agreement", {}),
        "total_test_samples": data.get("total_test_samples", 0)
    }


@app.get("/failures")
def get_failures():
    eval_file = "eval/eval_results.json"
    if not os.path.exists(eval_file):
        run_full_evaluation(output_path=eval_file)

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"failures": data.get("failure_analysis", [])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
