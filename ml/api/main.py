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

    # Log feature vector for debugging
    feature_names = [
        "estimated_monthly_income",
        "last_inflow_amount",
        "days_since_last_inflow",
        "credit_score",
        "total_spend_30d",
        "total_spend_90d",
        "transaction_count_30d",
        "transaction_count_90d",
        "avg_txn_amount_30d",
        "avg_txn_amount_90d",
        "max_txn_amount_90d",
        "txn_amount_median_90d",
        "spend_volatility_30d",
        "spend_volatility_90d",
        "spend_to_income_ratio_30d",
        "spend_to_income_ratio_90d",
        "avg_txn_over_income_ratio_90d",
        "txn_count_30d_norm",
        "current_txn_amount",
        "current_txn_mcc"
    ]
    
    print("\n" + "=" * 80)
    print("FEATURE VECTOR TRANSFORMATION")
    print("=" * 80)
    for name, value in zip(feature_names, feature_vector):
        print(f"{name:35s}: {value:>15.4f}")
    print("=" * 80 + "\n")

    dataset = ml_pipeline["dataset"]
    scaler_stats = ml_pipeline["scaler_stats"]
    vae = ml_pipeline["vae"]

    input_tensor = dataset.prepare_input_list(feature_vector, scaler_stats)
    decisions = vae.anomaly_score(input_tensor)
    decision = decisions[0]

    anomaly_score = decision["score"]
    estimated_monthly_income = feature_vector[0]
    
    epsilon_value = decision.get('epsilon', None)
    epsilon_display = f"{epsilon_value:>15.4f}" if epsilon_value is not None else f"{'N/A':>15s}"
    
    print("=" * 80)
    print("ANOMALY DETECTION RESULT")
    print("=" * 80)
    print(f"Anomaly Score:                      {anomaly_score:>15.4f}")
    print(f"Risk Level:                         {decision['risk']:>15s}")
    print(f"Reconstruction Error:               {decision['reconstruction_error']:>15.4f}")
    print(f"KL Divergence:                      {decision['kl_divergence']:>15.4f}")
    print(f"Normalized Score (epsilon):         {epsilon_display}")
    print("=" * 80 + "\n")
    
    n = max_installments(
        monthly_income=estimated_monthly_income,
        transaction_amount=request.transaction_amount,
        anomaly_score=anomaly_score
    )
    
    print("=" * 80)
    print("INSTALLMENT CALCULATION")
    print("=" * 80)
    print(f"Monthly Income:                     £{estimated_monthly_income:>14.2f}")
    print(f"Transaction Amount:                 £{request.transaction_amount:>14.2f}")
    print(f"Affordability Ratio:                 {abs(request.transaction_amount) / (estimated_monthly_income + 1):>15.4f}")
    print(f"Final Installments:                  {n:>15d}")
    print(f"Decision:                           {'APPROVED' if n > 0 else 'DECLINED':>15s}")
    print("=" * 80 + "\n")

    return PredictResponse(
        approved=n > 0,
        max_installments=n
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

