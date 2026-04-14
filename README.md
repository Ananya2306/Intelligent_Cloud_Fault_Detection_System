# 📘 Intelligent Cloud Fault Detection System  
*A Hybrid Machine Learning-Based Approach for Predictive Infrastructure Monitoring*

---

## 📌 Abstract

This project presents an intelligent fault detection system designed to identify anomalies and potential failures in cloud infrastructure environments. Leveraging concepts inspired by large-scale distributed systems, the solution integrates machine learning techniques with rule-based logic to enhance prediction reliability. The system is implemented as a full-stack application, combining backend model inference using FastAPI with an interactive frontend dashboard for real-time visualization. The objective is to enable early fault detection and proactive system monitoring.

---

## 📌 1. Introduction

Cloud infrastructure operates at massive scale, executing thousands of tasks concurrently. Monitoring such systems is critical, as failures can lead to performance degradation, service downtime, or resource wastage.

Traditional monitoring relies on static threshold-based alerts, which lack adaptability and intelligence. This project introduces a hybrid approach combining:

- Machine Learning-based fault prediction  
- Rule-based validation for critical scenarios  
- Interactive visualization for monitoring  

---

## 📌 2. Problem Statement

> To design a system capable of detecting potential faults in cloud environments using system-level metrics such as CPU usage, memory utilization, and workload intensity.

---

## 📌 3. Dataset Description

The dataset used in this project is inspired by the **Google Cluster Trace Dataset**, representing real-world cloud infrastructure behavior.

### Features Used:
- CPU Utilization  
- Memory Usage  
- Maximum Resource Usage  

> Note: Due to size constraints, the raw dataset is not included in this repository.

---

## 📌 4. Methodology

The system follows a structured machine learning pipeline:

---

### 🔹 4.1 Data Preprocessing

Steps performed:

- Removal of irrelevant columns  
- Handling missing values using forward-fill  
- Extraction of structured data from raw fields  
- Selection of key features (CPU, Memory, Max Usage)  

Output:
```
data/processed/cleaned_data.csv
```

---

### 🔹 4.2 Feature Engineering

- Normalization of numerical features  
- Scaling using MinMaxScaler  
- Transformation into model-compatible format  

---

### 🔹 4.3 Model Development

#### ✅ Random Forest (Primary Model)
- Supervised classification  
- Handles non-linear relationships  
- Provides probability-based confidence  

#### ✅ Isolation Forest
- Unsupervised anomaly detection  
- Detects abnormal system behavior  

#### ✅ LSTM (Experimental)
- Captures temporal patterns  
- Used for sequence-based learning (future scope)

---

### 🔹 4.4 Model Training

- Dataset splitting (train/test)  
- Training using Scikit-learn  
- Evaluation using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

---

### 🔹 4.5 Hybrid Decision Logic

To improve reliability, rule-based conditions were integrated:

```python
if cpu > 0.75 and memory > 0.75:
    final_pred = 1
elif cpu > 0.85:
    final_pred = 1
else:
    final_pred = model_pred
```

This ensures:
- Critical system conditions are always flagged
- ML predictions are validated

---

###  4.6 Confidence Estimation

Model confidence is calculated using:
```
rf.predict_proba()
```

Confidence = Maximum probability score

## 📌 5. System Architecture
```
Frontend (Netlify)
        ↓
FastAPI Backend (Render)
        ↓
Machine Learning Model (.pkl)
        ↓
Prediction + Confidence
        ↓
Frontend Visualization
```

---

## 📌 6. Implementation Details
Backend:
```
FastAPI
Scikit-learn
NumPy
Joblib
```

Frontend:
```
HTML
JavaScript
Chart.js
```

---

## 📌 7. API Design
Endpoint:
```
POST /predict
```

Input:
```
{
  "cpu": 0.8,
  "memory": 0.7,
  "max_usage": 0.9
}
```

Output:
```
{
  "prediction": 1,
  "confidence": 0.87
}
```

---

## 📌 8. Frontend Features

- Interactive sliders for system metrics
- Real-time fault prediction
- Ensemble-based verdict display
- Confidence visualization
- Model comparison view

---

## 📌 9. Evaluation Metrics
```
Metric	Value
Accuracy: 0.9879 (~98.79%)
Precision: 1.00
Recall: 0.9472
F1 Score: 0.9729
```

---

## 📌 10. Challenges Faced

- Handling large-scale dataset
- Parsing complex data formats
- GitHub large file limitations
- Deployment compatibility issues (TensorFlow vs Python version)
- Frontend-backend communication errors

---

## 📌 11. Solutions Implemented

- Dataset removal + external referencing
- Git history cleanup
- Removal of TensorFlow for deployment
- Python version control (3.11)
- CORS integration in FastAPI
- API debugging using /docs

---

## 📌 12. Deployment
Backend:
```
Platform: Render
Framework: FastAPI
```
Frontend:
```
Platform: Netlify
Static deployment
``
Live Architecture:
```
User → Netlify UI → Render API → ML Model → Response
```

---

## 📌 13. Future Work

- Real-time streaming data integration
- Deep learning-based anomaly detection
- Alert system (email/SMS)
- Docker + Kubernetes deployment
- Explainable AI (SHAP, LIME)

---

## 📌 14. Conclusion

This project demonstrates a complete AI-powered fault detection system integrating data preprocessing, machine learning, backend APIs, and frontend visualization. The hybrid approach enhances prediction reliability and enables proactive monitoring of cloud systems.

---

## 📌 15. How to Run Locally
Backend:
```
uvicorn backend.app:app --reload
```

Frontend:
```
cd frontend
python -m http.server 5500
```
Open:
```
http://127.0.0.1:5500
```

---

📌 16. Project Structure
```
MINOR/
│
├── backend/
│   └── app.py
├── frontend/
│   └── index.html
├── models/
│   ├── random_forest.pkl
│   ├── lstm_model.h5
│   ├── isolation_forest.pkl
├── data/
├── src/
├── outputs/
├── requirements.txt
```

---

