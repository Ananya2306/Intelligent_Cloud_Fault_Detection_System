"""
visualize.py — Generate evaluation plots and save to outputs/graphs/.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "outputs/graphs"

def plot(df, metrics: dict = None):
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = df.head(2000).copy()

    # ── 1. Fault timeline ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Cloud Fault Detection — System Overview", fontsize=14, fontweight='bold')

    for ax, col, color, label in zip(axes,
            ['cpu', 'memory', 'max_usage'],
            ['#2196F3', '#4CAF50', '#FF9800'],
            ['CPU Utilization', 'Memory Usage', 'Max Resource Usage']):
        ax.plot(sample[col], color=color, linewidth=0.8, alpha=0.85)
        fault_idx = sample.index[sample['rf_pred'] == 1]
        ax.scatter(fault_idx - sample.index[0],
                   sample.loc[fault_idx, col],
                   c='#F44336', s=12, zorder=5, label='Fault')
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.axhline(0.75, color='orange', linestyle='--', linewidth=0.6, alpha=0.6)
        ax.axhline(0.85, color='red',    linestyle='--', linewidth=0.6, alpha=0.6)

    axes[-1].set_xlabel("Sample Index", fontsize=9)
    red_patch = mpatches.Patch(color='#F44336', label='Predicted Fault')
    fig.legend(handles=[red_patch], loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fault_timeline.png"), dpi=150)
    plt.close()
    print(f"[viz] Saved fault_timeline.png")

    # ── 2. Model comparison bar ──────────────────────────────────────────────
    if metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        vals    = [metrics.get(k, 0)*100 for k in ['accuracy','precision','recall','f1','auc']]
        colors  = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']
        bars    = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(80, 105)
        ax.set_title('Random Forest — Performance Metrics', fontweight='bold')
        ax.set_ylabel('Score (%)')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "metrics_bar.png"), dpi=150)
        plt.close()
        print(f"[viz] Saved metrics_bar.png")

    # ── 3. Confusion matrix heatmap ──────────────────────────────────────────
    if metrics and 'cm' in metrics:
        cm  = metrics['cm']
        fig, ax = plt.subplots(figsize=(5, 4))
        im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        ticks = ['Normal', 'Fault']
        ax.set_xticks([0, 1]); ax.set_xticklabels(ticks)
        ax.set_yticks([0, 1]); ax.set_yticklabels(ticks)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — Random Forest', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
        plt.close()
        print(f"[viz] Saved confusion_matrix.png")

    # ── 4. Feature importance ────────────────────────────────────────────────
    try:
        import joblib
        rf = joblib.load("models/random_forest.pkl")
        feats = ['cpu', 'memory', 'max_usage', 'cpu_mean', 'memory_mean',
                 'cpu_std', 'memory_std', 'resource_pressure',
                 'cpu_delta', 'memory_delta']
        if len(rf.feature_importances_) == len(feats):
            fig, ax = plt.subplots(figsize=(8, 4))
            idx = np.argsort(rf.feature_importances_)[::-1]
            ax.bar([feats[i] for i in idx], rf.feature_importances_[idx],
                   color='#2196F3', edgecolor='white')
            ax.set_title('Feature Importance — Random Forest (Gini)', fontweight='bold')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=30, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
            plt.close()
            print(f"[viz] Saved feature_importance.png")
    except Exception:
        pass

    print("[viz] All plots saved.")
