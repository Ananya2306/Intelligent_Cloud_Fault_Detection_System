import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast, os, joblib

SCALER_PATH = "models/scaler.pkl"

def extract_cpu(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return float(value.get('cpus', 0))
    except:
        return 0.0

def extract_memory(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return float(value.get('memory', 0))
    except:
        return 0.0

def preprocess_google_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print(f"[preprocess] Original shape: {df.shape}")

    # Parse nested JSON-like fields
    df['cpu']       = df['average_usage'].apply(extract_cpu)
    df['memory']    = df['average_usage'].apply(extract_memory)
    df['max_usage'] = df['maximum_usage'].apply(extract_cpu)

    # Retain only relevant columns
    df = df[['time', 'cpu', 'memory', 'max_usage', 'failed']].copy()
    df.rename(columns={'failed': 'fault'}, inplace=True)

    # Impute missing values
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # Min-Max normalisation
    scaler = MinMaxScaler()
    df[['cpu', 'memory', 'max_usage']] = scaler.fit_transform(
        df[['cpu', 'memory', 'max_usage']]
    )
    joblib.dump(scaler, SCALER_PATH)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Saved cleaned data → {output_path}  shape: {df.shape}")
    return df
