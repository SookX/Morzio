# Morzio - Buy Now, Pay Later Platform

A complete fintech payment platform that enables merchants to offer installment payment options to customers, with AI-powered risk assessment to determine eligibility and payment terms.

## 🎯 The Idea

**Morzio** is a **Buy Now, Pay Later (BNPL)** platform that allows customers to split purchases into installments (4-44 payments) instead of paying upfront. Unlike traditional credit checks, Morzio uses **machine learning** to assess risk in real-time by analyzing transaction history patterns.

### Key Features

- **Instant Approval** - No credit bureau checks, decisions in seconds
- **Flexible Terms** - 4 to 44 installments based on risk and affordability
- **Real-time Risk Assessment** - ML model analyzes 90 days of transaction history
- **Merchant Integration** - Android POS app for easy merchant onboarding
- **Bank Integration** - Plaid API for secure transaction data access

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  Android POS    │  Merchant enters amount
│  (Terminal)     │  → Generates QR code
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Spring Boot    │  Creates payment session
│  Server (Java)  │  → Links to Plaid
└────────┬────────┘
         │
         ├──► Plaid API (Bank data)
         │
         ▼
┌─────────────────┐
│  FastAPI ML    │  Analyzes transactions
│  Service (Py)   │  → Returns installments
└─────────────────┘
```

### Three Main Components

1. **Terminal** - Android POS app for merchants
2. **Server** - Spring Boot backend handling business logic
3. **ML Service** - Python FastAPI for risk assessment

---

## 🔄 Complete Payment Flow

### Step 1: Merchant Initiates Payment

**Location:** Android POS App (`terminal/app/`)

1. Merchant enters purchase amount (e.g., £250.00)
2. App calls `POST /api/payment/initiate` with amount
3. Server creates `PaymentSession` with status `PENDING`
4. Server returns payment URL: `http://morzio.com/pay/{sessionId}`
5. App generates QR code from URL
6. Customer scans QR code with their phone

**Code:** `terminal/app/src/main/java/com/morzio/pos/viewmodels/QRCodeViewModel.kt`

### Step 2: Customer Opens Payment Page

**Location:** Web Browser (Thymeleaf template)

1. Customer scans QR → Opens `http://morzio.com/pay/{sessionId}`
2. Server renders `payment.html` with:
   - Purchase amount
   - Plaid Link token (for bank connection)
3. Customer clicks "Connect Bank" → Plaid Link opens
4. Customer authenticates with their bank

**Code:** `server/src/main/java/com/morzio/server/controllers/PaymentsController/PaymentPageController.java`

### Step 3: Bank Data Collection

**Location:** Plaid API Integration

1. Customer authorizes Plaid to access bank account
2. Plaid returns `publicToken`
3. Server exchanges `publicToken` → `accessToken`
4. Server fetches:
   - **90 days of transactions** (amounts, dates, categories)
   - **Account balances**
5. Server calls ML service with transaction data

**Code:** `server/src/main/java/com/morzio/server/services/PlaidService.java`

### Step 4: ML Risk Assessment

**Location:** FastAPI ML Service (`ml/api/`)

**Input:**
```json
{
  "transactions": [
    {"amount": -2500.0, "date": "2025-11-01", "category": "INCOME"},
    {"amount": 45.0, "date": "2025-11-28", "category": "FOOD"},
    ... // up to 500 transactions
  ],
  "transaction_amount": 250.0,
  "transaction_mcc": 5411
}
```

**Process:**

1. **Feature Engineering** (`ml/api/feature_builder.py`)
   - Converts 90 days of transactions → **20-feature vector**
   - Calculates:
     - Income: Sum of negative amounts (inflows) / 3 months
     - Spending: Aggregates (30d, 90d windows)
     - Ratios: Spend-to-income, volatility, frequency
     - Uses default `credit_score = 715` (median from training data)

2. **Anomaly Detection** (`ml/vae_annomaly_detection/`)
   - Loads pre-trained **Variational Autoencoder (VAE)** model
   - Model reconstructs input → Measures reconstruction error
   - High error = anomalous spending patterns
   - Returns anomaly score (0-100+)

3. **Installment Calculation** (`ml/dema_revenue/installment_policy.py`)
   - Formula uses 3 inputs:
     - Monthly income (I)
     - Transaction amount (A)
     - Anomaly score (e)
   - Calculates:
     - **Affordability ratio**: ρ = A / (I + 1)
     - **Affordability factor**: f(ρ) = 1 - σ(k_f × (ρ - ρ₀))
     - **Anomaly weight**: w(ρ) = σ(k_a × (ρ - ρ₁))
     - **Anomaly factor**: a(e,ρ) = 1 - w(ρ) × e²
     - **Final**: N = 48 × f(ρ) × a(e,ρ), rounded to multiple of 4
   - **Decline logic**: If e > 0.90 AND ρ > 0.2 → 0 installments

**Output:**
```json
{
  "approved": true,
  "max_installments": 36
}
```

**Code:** `ml/api/main.py`

### Step 5: Response to Customer

**Location:** Server → Web Page

1. Server receives ML response
2. If `approved = true`:
   - Shows installment options (e.g., "Pay in 36 installments")
   - Customer selects plan
   - Creates `InstallmentPlan` entity
3. If `approved = false`:
   - Shows "Payment declined" message

**Code:** `server/src/main/java/com/morzio/server/controllers/PlaidController/PlaidApiController.java`

---

## 🧠 Machine Learning Pipeline

### Model Architecture

**Variational Autoencoder (VAE)** for anomaly detection:

```
Input (20 features)
    ↓
Encoder (3 layers, SiLU activation)
    ↓
Latent Space (μ, σ) → Sample z
    ↓
Decoder (3 layers, SiLU activation)
    ↓
Reconstruction
```

**Training:**
- **Dataset:** 44,615 legitimate transaction feature vectors
- **Epochs:** 100 (with early stopping)
- **Loss:** Reconstruction Error + KL Divergence
- **Checkpoint:** `ml/vae_annomaly_detection/checkpoints/best_model.h5`

### Feature Vector (20 Features)

| # | Feature | Source | Example |
|---|---------|--------|---------|
| 1 | `estimated_monthly_income` | Plaid income txns / 3 | £2,000 |
| 2 | `last_inflow_amount` | Most recent income | £2,500 |
| 3 | `days_since_last_inflow` | Days since income | 5 |
| 4 | `credit_score` | Default value | 715 |
| 5-14 | Spending aggregates | Plaid expense txns | 30d/90d windows |
| 15-18 | Ratios & normalizations | Calculated | Spend/income ratios |
| 19 | `current_txn_amount` | POS | -£250 |
| 20 | `current_txn_mcc` | POS | 5411 (grocery) |

### Installment Formula Details

**Sigmoid-based smooth transitions** (no hard cutoffs):

- **ρ₀ = 0.45** - Affordability threshold
- **ρ₁ = 0.5** - Anomaly weight threshold
- **k_f = 5.0, k_a = 5.0** - Sigmoid steepness
- **Max installments = 48** (rounded to multiples of 4)

**Behavior:**
- Small purchase + normal user → **44 installments**
- Medium purchase + normal user → **32-40 installments**
- Large purchase + risky user → **DECLINED (0)**

---

## 📁 Project Structure

```
Morzio-1/
├── terminal/                    # Android POS App
│   └── app/
│       ├── MainActivity.kt      # Entry point
│       ├── ui/screens/          # Compose screens
│       ├── viewmodels/          # MVVM ViewModels
│       └── data/api/            # REST client
│
├── server/                      # Spring Boot Backend
│   └── src/main/java/
│       ├── controllers/
│       │   ├── PaymentController.java      # Payment initiation
│       │   ├── PaymentPageController.java  # Web page rendering
│       │   └── PlaidApiController.java     # Plaid callbacks
│       ├── services/
│       │   ├── PaymentService.java         # Payment logic
│       │   └── PlaidService.java           # Plaid integration
│       ├── entities/
│       │   ├── PaymentSession.java         # Payment state
│       │   ├── InstallmentPlan.java        # Installment terms
│       │   └── Installment.java            # Individual payments
│       └── repositorys/                     # JPA repositories
│
└── ml/                          # Machine Learning
    ├── api/                     # FastAPI Service
    │   ├── main.py              # API endpoints
    │   ├── feature_builder.py  # Plaid → feature vector
    │   └── requirements.txt
    ├── vae_annomaly_detection/  # VAE Model
    │   ├── model/                # VAE architecture
    │   ├── dataset/            # Data loading
    │   ├── pipeline.py          # Training/inference
    │   ├── run.py              # Training script
    │   ├── inference.py         # Inference script
    │   └── checkpoints/         # Saved models
    ├── dema_revenue/            # Feature Engineering
    │   ├── installment_policy.py  # Installment formula
    │   ├── build_feature_vector.py # Single client features
    │   └── build_training_dataset.py # Batch processing
    └── data/
        └── training_features.csv  # Training dataset
```

---

## 🚀 Getting Started

### Prerequisites

- Java 17+
- Python 3.12+
- Android Studio (for terminal app)
- PostgreSQL (for server)
- Plaid API keys

### 1. ML Service

```bash
cd ml
source vae_annomaly_detection/venv/bin/activate
pip install -r api/requirements.txt
python -m api.main
```

Runs on `http://localhost:8000`

### 2. Server

```bash
cd server
# Configure application.properties with DB and Plaid keys
./mvnw spring-boot:run
```

Runs on `http://localhost:8080`

### 3. Terminal (Android)

1. Open `terminal/` in Android Studio
2. Sync Gradle
3. Run on device/emulator

---

## 🔌 API Endpoints

### ML Service (`http://localhost:8000`)

**POST** `/api/predict`
- Accepts Plaid transactions
- Returns `{approved: bool, max_installments: int}`

**GET** `/health`
- Health check

### Server (`http://localhost:8080`)

**POST** `/api/payment/initiate`
- Creates payment session
- Returns QR code URL

**GET** `/pay/{sessionId}`
- Renders payment page with Plaid Link

**POST** `/api/plaid/on-success`
- Handles Plaid callback
- Calls ML service
- Returns installment options

---

## 🧪 Testing

### ML Service Test

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": -2500.0, "date": "2025-11-01", "category": "INCOME"},
      {"amount": 45.0, "date": "2025-11-28", "category": "FOOD"}
    ],
    "transaction_amount": 250.0,
    "transaction_mcc": 5411
  }'
```

### Expected Response

```json
{
  "approved": true,
  "max_installments": 36
}
```

---

## 📊 ML Model Performance

**Training Data:**
- 44,615 samples
- 20 features per sample
- 100 epochs training

**Baseline Scores (on training data):**
- Mean anomaly: 1.48
- Median: 1.37
- 99th percentile: 2.86
- Max: 6.52

**Test Results:**
- Normal users: 1.0-2.0 score → 36-44 installments ✅
- Risky users: 15+ score → DECLINED ✅

---

## 🔐 Security Considerations

- **Plaid tokens** - Stored securely, never exposed
- **Rate limiting** - Implemented on server endpoints
- **Data privacy** - Transaction data only used for risk assessment
- **Model security** - Checkpoint files committed (consider encryption for production)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Terminal** | Kotlin, Jetpack Compose, Retrofit |
| **Server** | Java 17, Spring Boot 4.0, Thymeleaf, PostgreSQL |
| **ML Service** | Python 3.12, FastAPI, PyTorch, Pandas |
| **ML Model** | Variational Autoencoder (VAE) |
| **Bank Integration** | Plaid API |

---

## 📝 Key Design Decisions

1. **Separate ML Service** - Python ecosystem (pandas, torch) can't run in JVM
2. **VAE for Anomaly Detection** - Learns normal patterns, flags deviations
3. **Sigmoid Formula** - Smooth transitions, no hard cutoffs
4. **90 Days of Data** - Balance between accuracy and latency
5. **Default Credit Score** - 715 (median) since Plaid doesn't provide it

---

## 🚧 Future Enhancements

- [ ] Real-time payment status updates
- [ ] Installment plan persistence
- [ ] Merchant dashboard
- [ ] Customer payment reminders
- [ ] Model retraining pipeline
- [ ] A/B testing for formula parameters

---

## 📄 License

Private - All rights reserved
