from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(data):
    y_true = data['fault']
    y_pred = data['rf_pred']

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))