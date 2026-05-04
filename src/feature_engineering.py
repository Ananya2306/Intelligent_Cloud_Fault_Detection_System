import pandas as pd
import numpy as np

WINDOW = 5

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal rolling features from raw normalised columns.
    These features capture short-term trend information, helping
    the model distinguish transient spikes from sustained faults.
    """
    # Rolling means
    df['cpu_mean']    = df['cpu'].rolling(WINDOW, min_periods=1).mean()
    df['memory_mean'] = df['memory'].rolling(WINDOW, min_periods=1).mean()

    # Rolling standard deviations (variance signal)
    df['cpu_std']     = df['cpu'].rolling(WINDOW, min_periods=1).std().fillna(0)
    df['memory_std']  = df['memory'].rolling(WINDOW, min_periods=1).std().fillna(0)

    # Interaction feature: combined resource pressure
    df['resource_pressure'] = (df['cpu'] + df['memory'] + df['max_usage']) / 3.0

    # Delta features (rate of change)
    df['cpu_delta']    = df['cpu'].diff().fillna(0)
    df['memory_delta'] = df['memory'].diff().fillna(0)

    df.fillna(0, inplace=True)
    print(f"[feature_eng] Features created. Columns: {list(df.columns)}")
    return df
