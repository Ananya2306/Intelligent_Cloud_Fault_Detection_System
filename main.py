from src.preprocessing import preprocess_google_data
from src.feature_engineering import create_features
from src.train_ml_models import train_models
from src.train_lstm import train_lstm
from src.predict import predict
from src.evaluate import evaluate
from src.visualize import plot

# Step 1
data = preprocess_google_data(
    "data/raw/cloud_data.csv",
    "data/processed/cleaned_data.csv"
)

# Step 2
data = create_features(data)

# Step 3
train_models(data)

# Step 4
train_lstm(data)

# Step 5
data = predict(data)

# Step 6
evaluate(data)

# Step 7
plot(data)

print("FULL ML + LSTM PIPELINE DONE 🚀")