from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib

def train_models(data):
    X = data[['cpu','memory','max_usage']]
    y = data['fault']

    # Random Forest
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X, y)
    joblib.dump(rf, "models/random_forest.pkl")

    # Isolation Forest
    iso = IsolationForest(contamination=0.1)
    iso.fit(X)
    joblib.dump(iso, "models/isolation_forest.pkl")

    print("ML models trained successfully")