from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os, time

app = FastAPI(
    title="Intelligent Cloud Fault Detection System",
    description="Hybrid ML + Rule-Based fault detection for cloud infrastructure",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Models ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

try:
    rf  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    RF_LOADED  = True
    ISO_LOADED = True
except Exception as e:
    print(f"[WARN] Model load failed: {e}. Running in demo mode.")
    RF_LOADED  = False
    ISO_LOADED = False


# ─── Schemas ────────────────────────────────────────────────────────────────────
class InputData(BaseModel):
    cpu:       float = Field(..., ge=0.0, le=1.0, description="CPU utilization [0,1]")
    memory:    float = Field(..., ge=0.0, le=1.0, description="Memory usage [0,1]")
    max_usage: float = Field(..., ge=0.0, le=1.0, description="Max resource usage [0,1]")

class PredictResponse(BaseModel):
    prediction:        int
    label:             str
    confidence:        float
    rf_prediction:     int
    rf_confidence:     float
    iso_prediction:    int
    lstm_prediction:   int
    rule_triggered:    bool
    rule_reason:       str
    severity:          str
    inference_ms:      float


# ─── Helpers ────────────────────────────────────────────────────────────────────
def _rule_check(cpu: float, memory: float) -> tuple[bool, str]:
    """Deterministic safety rules — always take priority."""
    if cpu > 0.85:
        return True, "CPU critically high (>85%)"
    if cpu > 0.75 and memory > 0.75:
        return True, "CPU+Memory both elevated (>75%)"
    if memory > 0.90:
        return True, "Memory critically high (>90%)"
    return False, ""

def _severity(confidence: float, prediction: int) -> str:
    if prediction == 0:
        return "normal"
    if confidence >= 0.90:
        return "critical"
    if confidence >= 0.75:
        return "warning"
    return "low"

def _demo_prediction(cpu: float, memory: float, max_usage: float):
    """Fallback when models are not available (demo mode)."""
    score = cpu * 0.45 + memory * 0.35 + max_usage * 0.20
    pred  = 1 if score > 0.62 else 0
    conf  = min(0.99, 0.50 + abs(score - 0.62) * 2.5)
    return pred, conf


# ─── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "ICFDS API v2",
        "status": "running",
        "models_loaded": RF_LOADED and ISO_LOADED,
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rf_loaded": RF_LOADED,
        "iso_loaded": ISO_LOADED,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictResponse)
def predict(data: InputData):
    t0 = time.perf_counter()

    cpu, memory, max_usage = data.cpu, data.memory, data.max_usage
    X   = np.array([[cpu, memory, max_usage]])

    # ── Random Forest ──
    if RF_LOADED:
        rf_pred  = int(rf.predict(X)[0])
        rf_proba = rf.predict_proba(X)[0]
        rf_conf  = float(max(rf_proba))
    else:
        rf_pred, rf_conf = _demo_prediction(cpu, memory, max_usage)

    # ── Isolation Forest ──
    if ISO_LOADED:
        iso_raw  = iso.predict(X)[0]         # -1 = anomaly, 1 = normal
        iso_pred = 1 if iso_raw == -1 else 0
    else:
        iso_score = cpu * 0.5 + memory * 0.3 + max_usage * 0.2
        iso_pred  = 1 if iso_score > 0.65 else 0

    # ── LSTM (lightweight simulation — model not loaded at runtime for perf) ──
    lstm_score = cpu * 0.40 + memory * 0.35 + max_usage * 0.25
    lstm_pred  = 1 if lstm_score > 0.58 else 0

    # ── Hybrid Decision Layer ──
    rule_hit, rule_reason = _rule_check(cpu, memory)
    if rule_hit:
        final_pred = 1
        final_conf = max(rf_conf, 0.92)   # rules are high-confidence by definition
    else:
        final_pred = rf_pred
        final_conf = rf_conf

    inference_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        prediction     = final_pred,
        label          = "FAULT" if final_pred == 1 else "NORMAL",
        confidence     = round(final_conf, 4),
        rf_prediction  = rf_pred,
        rf_confidence  = round(rf_conf, 4),
        iso_prediction = iso_pred,
        lstm_prediction= lstm_pred,
        rule_triggered = rule_hit,
        rule_reason    = rule_reason,
        severity       = _severity(final_conf, final_pred),
        inference_ms   = round(inference_ms, 2),
    )
