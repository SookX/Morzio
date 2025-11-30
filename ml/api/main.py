import sys
from pathlib import Path
from datetime import date
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

base_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str(base_path / "vae_annomaly_detection"))

from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import TransactionDataset
from utils import read_config
from dema_revenue.installment_policy import max_installments
from api.feature_builder import build_feature_vector, Transaction


class TransactionInput(BaseModel):
    amount: float
    date: date
    category: Optional[str] = None


class PredictRequest(BaseModel):
    transactions: List[TransactionInput]
    transaction_amount: float
    transaction_mcc: int = 5411


class PredictResponse(BaseModel):
    approved: bool
    max_installments: int
    risk_level: str
    anomaly_score: float
    feature_vector: Optional[List[float]] = None


ml_pipeline = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_path = Path(__file__).resolve().parent.parent
    config_path = base_path / "vae_annomaly_detection" / "config.yaml"
    data_path = base_path / "data" / "training_features.csv"
    checkpoint_path = base_path / "vae_annomaly_detection" / "checkpoints" / "best_model.h5"

    config = read_config(str(config_path))
    dataset = TransactionDataset(str(data_path))

    vae = VAEAnomalyDetectionPipeline(
        input_dim=dataset.input_dim,
        hidden_dim=int(config['model']['hidden_dim']),
        latent_dim=int(config['model']['latent_dim']),
        config=config
    )
    vae.load_model(str(checkpoint_path))

    ml_pipeline["vae"] = vae
    ml_pipeline["dataset"] = dataset
    ml_pipeline["scaler_stats"] = dataset.scaler_stats()

    print("ML Pipeline loaded successfully")
    yield
    ml_pipeline.clear()


app = FastAPI(
    title="Morzio ML API",
    description="Anomaly detection and installment prediction",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "vae" in ml_pipeline}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if "vae" not in ml_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    transactions = [
        Transaction(
            amount=t.amount,
            date=t.date,
            category=t.category
        )
        for t in request.transactions
    ]

    feature_vector = build_feature_vector(
        transactions=transactions,
        current_txn_amount=request.transaction_amount,
        current_txn_mcc=request.transaction_mcc,
    )

    dataset = ml_pipeline["dataset"]
    scaler_stats = ml_pipeline["scaler_stats"]
    vae = ml_pipeline["vae"]

    input_tensor = dataset.prepare_input_list(feature_vector, scaler_stats)
    decisions = vae.anomaly_score(input_tensor)
    decision = decisions[0]

    anomaly_score = decision["score"]
    risk_level = decision["risk"]

    estimated_monthly_income = feature_vector[0]
    n = max_installments(
        monthly_income=estimated_monthly_income,
        transaction_amount=request.transaction_amount,
        anomaly_score=anomaly_score
    )

    return PredictResponse(
        approved=n > 0,
        max_installments=n,
        risk_level=risk_level,
        anomaly_score=round(anomaly_score, 4),
        feature_vector=feature_vector
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

