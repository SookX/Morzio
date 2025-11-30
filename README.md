# Morzio - Buy Now, Pay Later Platform

A fintech payment platform that provides installment payment options with ML-powered risk assessment.

## Architecture

- **Terminal** - Android POS app (Kotlin/Compose)
- **Server** - Spring Boot backend (Java)
- **ML Service** - FastAPI anomaly detection & installment prediction (Python)

## Quick Start

### ML Service

```bash
cd ml
source vae_annomaly_detection/venv/bin/activate
pip install -r api/requirements.txt
python -m api.main
```

Runs on `http://localhost:8000`

### API Endpoint

**POST** `/api/predict`

```json
{
  "transactions": [
    {"amount": -2500.0, "date": "2025-11-01", "category": "INCOME"},
    {"amount": 45.0, "date": "2025-11-28", "category": "FOOD"}
  ],
  "transaction_amount": 250.0,
  "transaction_mcc": 5411
}
```

**Response:**
```json
{
  "approved": true,
  "max_installments": 36
}
```

### Server

```bash
cd server
./mvnw spring-boot:run
```

### Terminal (Android)

Open in Android Studio and run on device/emulator.

## ML Components

- **VAE Anomaly Detection** - Detects suspicious transaction patterns
- **Installment Policy** - Calculates max installments (0-44) based on income, purchase amount, and risk score

## License

Private

