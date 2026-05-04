# ⚡ Intelligent Cloud Fault Detection System (ICFDS)
*Hybrid ML + Rule-Based Predictive Infrastructure Monitoring*

**Authors:** Ananya · Kashish Aggarwal · Naina · Mridul Sharma · Khalid Raza Khan  
**Institution:** IILM University, Greater Noida — Department of CSE (AI & ML)  
**Project Code:** BTP2 CSE329

---

## Overview

ICFDS is a full-stack, production-grade cloud fault detection system that combines:

| Component | Description |
|-----------|-------------|
| **Random Forest** | Primary supervised classifier — 98.79% accuracy |
| **Isolation Forest** | Unsupervised anomaly detector — 85.30% accuracy |
| **LSTM (Experimental)** | Temporal deep learning — 92.10% accuracy |
| **Rule-Based Layer** | Deterministic override for critical conditions |
| **FastAPI Backend** | REST API deployed on Render |
| **Interactive Dashboard** | Real-time SPA deployed on Netlify |

---

## Project Structure

```
icfds/
├── backend/
│   └── app.py              FastAPI REST API with hybrid decision logic
├── frontend/
│   └── index.html          Full-stack dashboard (Chart.js + vanilla JS)
├── src/
│   ├── preprocessing.py    Data parsing, normalisation, imputation
│   ├── feature_engineering.py  Rolling means, std, deltas, pressure index
│   ├── train_ml_models.py  Random Forest + Isolation Forest training
│   ├── train_lstm.py       Sliding-window LSTM with early stopping
│   ├── predict.py          Batch inference + hybrid rule layer
│   ├── evaluate.py         Full metrics report (accuracy, F1, AUC, CM)
│   └── visualize.py        Fault timeline, metrics bar, confusion matrix, feature importance
├── models/                 Trained .pkl and .h5 model files
├── data/
│   ├── raw/                Place cloud_data.csv here
│   └── processed/          Cleaned + normalised output
├── outputs/graphs/         Generated evaluation plots
├── main.py                 Full pipeline runner
├── requirements.txt
├── render.yaml             Render deployment config
└── runtime.txt             Python 3.11
```

---

## Quick Start

### 1 — Backend (FastAPI)

```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

API runs at `http://localhost:8000`

```bash
# Test endpoint
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"cpu": 0.82, "memory": 0.78, "max_usage": 0.90}'
```

Response:
```json
{
  "prediction": 1,
  "label": "FAULT",
  "confidence": 0.9412,
  "rf_prediction": 1,
  "rf_confidence": 0.9412,
  "iso_prediction": 1,
  "lstm_prediction": 1,
  "rule_triggered": true,
  "rule_reason": "CPU+Memory both elevated (>75%)",
  "severity": "critical",
  "inference_ms": 4.2
}
```

### 2 — Full ML Pipeline (requires dataset)

```bash
# Place Google Cluster Trace CSV at data/raw/cloud_data.csv
python main.py
```

### 3 — Frontend

```bash
cd frontend
python -m http.server 5500
# Open http://localhost:5500
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Fault prediction |
| GET | `/docs` | Auto-generated OpenAPI docs |

### POST /predict

**Input:**
```json
{"cpu": 0.0-1.0, "memory": 0.0-1.0, "max_usage": 0.0-1.0}
```

**Output fields:** `prediction`, `label`, `confidence`, `rf_prediction`, `rf_confidence`, `iso_prediction`, `lstm_prediction`, `rule_triggered`, `rule_reason`, `severity`, `inference_ms`

---

## Hybrid Decision Logic

```python
if cpu > 0.85:
    return FAULT  # CPU critically high
elif cpu > 0.75 and memory > 0.75:
    return FAULT  # Resource saturation
elif memory > 0.90:
    return FAULT  # Memory critically high
else:
    return rf_prediction  # Defer to ML model
```

This guarantees zero false negatives for safety-critical conditions.

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Random Forest** | **98.79%** | **100%** | **94.72%** | **97.29%** | **0.974** |
| Isolation Forest | 85.30% | 82.50% | 80.20% | 81.33% | 0.881 |
| LSTM (Experimental) | 92.10% | 90.40% | 88.70% | 89.54% | 0.934 |

---

## Deployment

### Backend — Render
1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. Use `render.yaml` — auto-deploys on push

### Frontend — Netlify
1. Drag & drop `frontend/` folder on [netlify.com](https://netlify.com)
2. Or `netlify deploy --dir frontend/`

---

## Dataset

**Google Cluster Trace v3** — production cluster telemetry over 29 days  
Available: https://github.com/google/cluster-data

Required columns: `time`, `average_usage` (nested JSON), `maximum_usage` (nested JSON), `failed`

---

## Future Work

- Real-time Kafka/WebSocket telemetry streaming
- Sliding-window LSTM (60-step sequences)
- SHAP explainability per prediction
- Docker + Kubernetes deployment
- Adaptive threshold learning

---

## Faculty Guide
Ms. Niharika Chaudhary — Department of CSE (AI & ML), IILM University
