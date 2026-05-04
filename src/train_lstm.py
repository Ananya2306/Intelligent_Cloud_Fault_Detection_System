import numpy as np
import os

WINDOW_SIZE = 10   # steps per sequence

def _build_sequences(X: np.ndarray, y: np.ndarray, window: int):
    """Slide a window over the time-series to build (N, window, features) tensors."""
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i : i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

def train_lstm(df, model_dir: str = "models", max_samples: int = 20_000):
    """
    Train a single-layer LSTM for temporal fault prediction.
    Uses a sliding window of WINDOW_SIZE steps over the normalised features.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("[LSTM] TensorFlow not available — skipping LSTM training.")
        return None

    os.makedirs(model_dir, exist_ok=True)

    # Subsample for speed
    df_s  = df.head(max_samples).copy()
    feats = ['cpu', 'memory', 'max_usage', 'cpu_mean', 'memory_mean']
    feats = [f for f in feats if f in df_s.columns]
    X = df_s[feats].values
    y = df_s['fault'].values

    X_seq, y_seq = _build_sequences(X, y, WINDOW_SIZE)

    split      = int(len(X_seq) * 0.80)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    print(f"[LSTM] Sequences: train={len(X_tr)}, test={len(X_te)}, features={X.shape[1]}")

    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh',
             input_shape=(WINDOW_SIZE, X.shape[1])),
        Dropout(0.20),
        LSTM(32, activation='tanh'),
        Dropout(0.20),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        epochs          = 15,
        batch_size      = 64,
        validation_data = (X_te, y_te),
        callbacks       = [es],
        verbose         = 1,
    )

    model.save(os.path.join(model_dir, "lstm_model.h5"))
    print(f"[LSTM] Saved → {model_dir}/lstm_model.h5")
    return model, history
