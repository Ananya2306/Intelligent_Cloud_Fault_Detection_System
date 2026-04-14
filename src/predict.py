import joblib
import numpy as np
from tensorflow.keras.models import load_model

def predict(data):
    X = data[['cpu','memory','max_usage']]

    rf = joblib.load("models/random_forest.pkl")
    iso = joblib.load("models/isolation_forest.pkl")
    lstm = load_model("models/lstm_model.h5")

    data['rf_pred'] = rf.predict(X)
    data['iso_pred'] = iso.predict(X)

    # LSTM prediction
    X_lstm = X.values.reshape((X.shape[0],1,X.shape[1]))
    data['lstm_pred'] = (lstm.predict(X_lstm) > 0.5).astype(int)

    print("Predictions done (ML + LSTM)")

    return data