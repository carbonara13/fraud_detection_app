from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from src.schemas import TransactionSchema, PredictionResponseSchema
from starlette.concurrency import run_in_threadpool
import joblib
from datetime import datetime, timezone
import pandas as pd
from uuid import uuid4

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load("models/model.joblib") # startup
    yield
    app.state.model = None # shutdown
    
def log_suspicious(tx_id: str, proba: float):
    with open("logs/suspicious_transactions.log", "a") as f:
        f.write(f"{tx_id},{proba}\n")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503,"Model is not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/v1/predict", response_model=PredictionResponseSchema)
async def predict(tx: TransactionSchema, request: Request, bg: BackgroundTasks):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503,"Model is not loaded")
    
    proba = await run_in_threadpool(model.predict_proba, tx.to_features())

    fraud_probability = proba[0][1]
    is_fraud = fraud_probability >= 0.5
    risk_level = "HIGH" if fraud_probability >= 0.7 else "MEDIUM" if fraud_probability >= 0.3 else "LOW"
    transaction_id = uuid4()
    
    # Log suspicious transactions in the background
    if risk_level == "HIGH":
        bg.add_task(log_suspicious, str(transaction_id), fraud_probability)
    
    return PredictionResponseSchema(
        transaction_id=transaction_id,
        timestamp=datetime.now(tz=timezone.utc),
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
        risk_level=risk_level
    )

@app.post("/v1/predict/batch", response_model=List[PredictionResponseSchema])
async def predict_batch(txs: List[TransactionSchema], request: Request, bg: BackgroundTasks):
    if len(txs) > 50:
        raise HTTPException(400, "Maximum 50 transactions per batch")

    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503, "Model is not loaded")

    probabilities = await run_in_threadpool(
        model.predict_proba, pd.DataFrame([tx.model_dump(mode="json") for tx in txs])
    )

    results = []
    for tx, proba in zip(txs, probabilities):
        fraud_probability = proba[1]
        is_fraud = fraud_probability >= 0.5
        risk_level = "HIGH" if fraud_probability >= 0.7 else "MEDIUM" if fraud_probability >= 0.3 else "LOW"
        transaction_id = uuid4()

        if risk_level == "HIGH":
            bg.add_task(log_suspicious, str(transaction_id), fraud_probability)

        results.append(
            PredictionResponseSchema(
                transaction_id=transaction_id,
                timestamp=datetime.now(tz=timezone.utc),
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_level=risk_level
            )
        )

    return results
