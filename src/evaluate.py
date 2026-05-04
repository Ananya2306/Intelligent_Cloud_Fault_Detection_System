"""
evaluate.py — Print and return a full evaluation report.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def evaluate(df: pd.DataFrame, pred_col: str = 'rf_pred') -> dict:
    y_true = df['fault']
    y_pred = df[pred_col]
    y_prob = df['rf_proba'] if 'rf_proba' in df.columns else y_pred.astype(float)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    cm    = confusion_matrix(y_true, y_pred)

    print("\n" + "="*50)
    print("  EVALUATION REPORT")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("\n  Full Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal','Fault']))
    print("="*50 + "\n")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc, cm=cm)
