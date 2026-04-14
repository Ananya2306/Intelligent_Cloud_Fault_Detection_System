from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
rf = joblib.load("models/random_forest.pkl")

class InputData(BaseModel):
    cpu: float
    memory: float
    max_usage: float
@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    cpu = data.cpu
    memory = data.memory
    max_usage = data.max_usage

    # ML model prediction
    X = [[cpu, memory, max_usage]]
    model_pred = rf.predict(X)[0]

    # 🔥 GET PROBABILITY (NEW)
    prob = rf.predict_proba(X)[0]
    confidence = max(prob)

    # 🔥 RULE-BASED LOGIC
    if cpu > 0.75 and memory > 0.75:
        final_pred = 1
    elif cpu > 0.85:
        final_pred = 1
    else:
        final_pred = model_pred

    return {
        "prediction": int(final_pred),
        "confidence": float(confidence)
    }
