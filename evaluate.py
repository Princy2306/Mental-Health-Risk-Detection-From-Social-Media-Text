"""
evaluate.py — Reusable evaluation utilities for mental health NLP project.

Used by: train.py, notebooks/03_baselines.ipynb, notebooks/04_distilbert.ipynb
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

LABEL_ORDER = ['low', 'moderate', 'high']
PALETTE     = {'low': '#1D9E75', 'moderate': '#BA7517', 'high': '#D85A30'}


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: List[str]) -> Dict:
    """Return a flat dict of classification metrics."""
    return {
        'accuracy'         : round(accuracy_score(y_true, y_pred), 4),
        'f1_macro'         : round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        'f1_weighted'      : round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        'precision_macro'  : round(precision_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        'recall_macro'     : round(recall_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        'per_class_f1'     : dict(zip(classes,
                                      f1_score(y_true, y_pred, average=None, zero_division=0).round(4))),
    }


def measure_inference_time(model: Any, X_sample, n_trials: int = 100) -> float:
    """Return mean inference time in milliseconds per single sample."""
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        model.predict(X_sample[:1])
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 3)


def cross_val_metrics(model: Any, X, y, cv: int = 5) -> Dict:
    """Run stratified k-fold CV and return mean ± std metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'accuracy']
    scores = cross_validate(model, X, y, cv=skf, scoring=scoring,
                            return_train_score=True, n_jobs=-1)
    return {
        'cv_f1_macro'       : round(scores['test_f1_macro'].mean(),    4),
        'cv_f1_macro_std'   : round(scores['test_f1_macro'].std(),     4),
        'cv_f1_weighted'    : round(scores['test_f1_weighted'].mean(),  4),
        'cv_precision_macro': round(scores['test_precision_macro'].mean(), 4),
        'cv_recall_macro'   : round(scores['test_recall_macro'].mean(), 4),
        'cv_accuracy'       : round(scores['test_accuracy'].mean(),    4),
        'train_f1_macro'    : round(scores['train_f1_macro'].mean(),   4),
        'overfit_gap'       : round(scores['train_f1_macro'].mean() -
                                    scores['test_f1_macro'].mean(),    4),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: List[str], title: str,
                          save_path: Optional[str] = None) -> None:
    """Plot a normalised + raw confusion matrix side by side."""
    cm_raw  = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cm, fmt, ttl in [
        (axes[0], cm_raw,  'd',    'Raw counts'),
        (axes[1], cm_norm, '.2f',  'Row-normalised'),
    ]:
        sns.heatmap(cm, annot=True, fmt=fmt, ax=ax,
                    xticklabels=classes, yticklabels=classes,
                    cmap='Blues', linewidths=0.5,
                    annot_kws={'size': 11})
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True',      fontsize=11)
        ax.set_title(ttl,          fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_results_table(results_df: pd.DataFrame,
                       save_path: Optional[str] = None) -> None:
    """
    Horizontal bar chart comparing models on F1-macro.
    results_df must have columns: model, f1_macro, inference_ms, model_size_kb
    """
    df = results_df.sort_values('f1_macro')
    colors = ['#7F77DD' if 'BERT' in m else '#1D9E75' for m in df['model']]

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, len(df) * 0.8 + 1)))

    # F1-macro
    bars = axes[0].barh(df['model'], df['f1_macro'], color=colors, alpha=0.88)
    for bar, val in zip(bars, df['f1_macro']):
        axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel('F1-macro')
    axes[0].set_title('F1-macro (higher = better)', fontweight='bold')

    # Inference time
    axes[1].barh(df['model'], df['inference_ms'], color='#BA7517', alpha=0.78)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[1].text(row['inference_ms'] + 0.1, i,
                     f"{row['inference_ms']:.1f} ms", va='center', fontsize=9)
    axes[1].set_xlabel('ms / sample')
    axes[1].set_title('Inference time (lower = better)', fontweight='bold')

    # Model size
    axes[2].barh(df['model'], df['model_size_kb'], color='#D85A30', alpha=0.78)
    for i, (_, row) in enumerate(df.iterrows()):
        axes[2].text(row['model_size_kb'] + 1, i,
                     f"{row['model_size_kb']:.0f} KB", va='center', fontsize=9)
    axes[2].set_xlabel('KB')
    axes[2].set_title('Model size (lower = better)', fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Model Comparison Dashboard', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_cv_comparison(cv_results: Dict[str, Dict],
                       save_path: Optional[str] = None) -> None:
    """Bar chart with error bars showing CV F1-macro across models."""
    models = list(cv_results.keys())
    means  = [cv_results[m]['cv_f1_macro']     for m in models]
    stds   = [cv_results[m]['cv_f1_macro_std'] for m in models]
    gaps   = [cv_results[m]['overfit_gap']      for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#1D9E75' if g < 0.05 else '#D85A30' for g in gaps]

    axes[0].barh(models, means, xerr=stds, color=colors,
                 alpha=0.85, capsize=5, error_kw={'elinewidth': 1.5})
    axes[0].set_xlabel('CV F1-macro')
    axes[0].set_title('Cross-validation F1-macro ± std\n(green = low overfit gap)', fontweight='bold')
    axes[0].set_xlim(0, 1.1)
    for i, (m, v) in enumerate(zip(models, means)):
        axes[0].text(v + stds[i] + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    axes[1].barh(models, gaps, color=['#1D9E75' if g < 0.05 else '#D85A30' for g in gaps], alpha=0.85)
    axes[1].axvline(0.05, color='black', linestyle='--', linewidth=0.8, label='0.05 threshold')
    axes[1].set_xlabel('Train F1 − Val F1')
    axes[1].set_title('Overfit gap (lower = better generalisation)', fontweight='bold')
    axes[1].legend(fontsize=9)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Cross-Validation Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


# ── Error analysis ─────────────────────────────────────────────────────────────

def get_misclassified(texts: List[str], y_true: np.ndarray,
                      y_pred: np.ndarray, classes: List[str],
                      n: int = 20) -> pd.DataFrame:
    """Return DataFrame of misclassified examples for manual inspection."""
    mask = y_true != y_pred
    indices = np.where(mask)[0]
    rows = []
    for i in indices[:n]:
        rows.append({
            'index'    : i,
            'text'     : texts[i][:200] + ('...' if len(texts[i]) > 200 else ''),
            'true'     : classes[y_true[i]],
            'predicted': classes[y_pred[i]],
            'error_type': f"{classes[y_true[i]]} → {classes[y_pred[i]]}",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Error type breakdown:\n{df['error_type'].value_counts()}\n")
    return df


if __name__ == '__main__':
    print("evaluate.py loaded — all utilities available.")
    print(f"Exported: compute_metrics, cross_val_metrics, measure_inference_time,")
    print(f"          plot_confusion_matrix, plot_results_table, plot_cv_comparison,")
    print(f"          get_misclassified")
