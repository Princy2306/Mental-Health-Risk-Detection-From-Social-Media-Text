"""
shap_utils.py — SHAP explainability utilities for mental health NLP project.

Two explainer tracks:
  1. TF-IDF track  → LinearExplainer → token-level importance (what words drove the prediction)
  2. Linguistic track → LinearExplainer → feature-level importance (negation, pronoun rate, etc.)

Used by:
  - notebooks/04_shap_fairness.ipynb
  - app/predict.py  (inference-time token highlights)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import warnings
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')

PALETTE = {'high': '#D85A30', 'moderate': '#BA7517', 'low': '#1D9E75'}


# ── Explainer builders ─────────────────────────────────────────────────────────

def build_tfidf_explainer(lr_model, X_background) -> shap.LinearExplainer:
    """
    Build LinearExplainer for a LogisticRegression trained on TF-IDF features.
    Uses interventional feature perturbation — correct for correlated text features.
    """
    bg_sample = shap.utils.sample(X_background, min(100, X_background.shape[0]),
                                  random_state=42)
    return shap.LinearExplainer(lr_model, bg_sample)


def build_linguistic_explainer(lr_model, X_ling_background) -> shap.LinearExplainer:
    """Build LinearExplainer for model trained on 24 linguistic features (dense)."""
    bg = X_ling_background[:min(100, len(X_ling_background))]
    return shap.LinearExplainer(lr_model, bg)


# ── Global importance ──────────────────────────────────────────────────────────

def global_token_importance(explainer,
                            X_test,
                            feature_names,
                            class_names,
                            top_n=20):

    shap_arr = np.array(explainer.shap_values(X_test))

    mean_abs = np.abs(shap_arr).mean(axis=0)
    mean_signed = shap_arr.mean(axis=0)

    top_idx = mean_abs.argsort()[-top_n:][::-1]

    df = pd.DataFrame({
        "feature": [feature_names[j] for j in top_idx],
        "mean_abs_shap": mean_abs[top_idx].round(6),
        "mean_shap": mean_signed[top_idx].round(6),
    })

    # Same importance for both classes in binary case
    return {cls: df.copy() for cls in class_names}


def global_linguistic_importance(explainer,
                                 X_ling_test,
                                 ling_feature_names,
                                 class_names,
                                 top_n=12):

    shap_arr = np.array(explainer.shap_values(X_ling_test))

    mean_abs = np.abs(shap_arr).mean(axis=0)
    mean_signed = shap_arr.mean(axis=0)

    top_idx = mean_abs.argsort()[-top_n:][::-1]

    df = pd.DataFrame({
        "feature": [ling_feature_names[j] for j in top_idx],
        "mean_abs_shap": mean_abs[top_idx].round(6),
        "mean_shap": mean_signed[top_idx].round(6),
    })

    return {cls: df.copy() for cls in class_names}


# ── Individual prediction waterfall ───────────────────────────────────────────

def individual_shap_breakdown(explainer,
                              X_single,
                              feature_names,
                              class_names,
                              predicted_class_idx=0,
                              top_n=10):
    """Return top feature contributions for one prediction."""

    shap_arr = np.array(explainer.shap_values(X_single))

    if shap_arr.ndim == 2:
        sv_pred = shap_arr[0]
    else:
        sv_pred = shap_arr[0, :, predicted_class_idx]

    top_idx = np.abs(sv_pred).argsort()[-top_n:][::-1]

    return pd.DataFrame({
        "feature": [feature_names[j] for j in top_idx],
        "shap_value": sv_pred[top_idx].round(6),
        "direction": [
            "↑ pushes toward" if v > 0 else "↓ pushes away"
            for v in sv_pred[top_idx]
        ]
    })


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_global_token_importance(importance_dict: Dict[str, pd.DataFrame],
                                  class_names: List[str],
                                  title: str = 'Global Token Importance (SHAP)',
                                  save_path: Optional[str] = None) -> None:
    """Horizontal bar chart: top tokens per class, colour-coded by direction."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5.5 * n_classes, 8))
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, class_names):
        df = importance_dict[cls].head(15)
        colors = ['#D85A30' if v > 0 else '#7F77DD' for v in df['mean_shap']]
        ax.barh(df['feature'][::-1], df['mean_abs_shap'][::-1],
                color=colors[::-1], alpha=0.88)
        ax.set_title(f'Class: {cls}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Mean |SHAP|')
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    pos_patch = mpatches.Patch(color='#D85A30', alpha=0.88, label='Pushes toward class')
    neg_patch = mpatches.Patch(color='#7F77DD', alpha=0.88, label='Pushes away from class')
    fig.legend(handles=[pos_patch, neg_patch], loc='upper center',
               ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_waterfall(breakdown_df: pd.DataFrame,
                   predicted_class: str,
                   text_preview: str,
                   base_value: float = 0.0,
                   save_path: Optional[str] = None) -> None:
    """Waterfall chart for a single prediction's SHAP breakdown."""
    df = breakdown_df.head(10).copy()
    df = df.sort_values('shap_value')

    colors = ['#D85A30' if v > 0 else '#7F77DD' for v in df['shap_value']]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df['feature'], df['shap_value'], color=colors, alpha=0.88)
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, df['shap_value']):
        x_pos = val + (0.002 if val >= 0 else -0.002)
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.4f}', va='center', ha=ha, fontsize=8.5)
    ax.set_xlabel('SHAP value (contribution to predicted class)')
    ax.set_title(f'Prediction breakdown → class: {predicted_class}\n'
                 f'"{text_preview[:80]}..."', fontweight='bold', fontsize=10)
    pos_p = mpatches.Patch(color='#D85A30', alpha=0.88, label='Increases confidence')
    neg_p = mpatches.Patch(color='#7F77DD', alpha=0.88, label='Decreases confidence')
    ax.legend(handles=[pos_p, neg_p], fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_linguistic_feature_importance(importance_dict: Dict[str, pd.DataFrame],
                                        class_names: List[str],
                                        save_path: Optional[str] = None) -> None:
    """Grouped bar chart comparing linguistic feature importance across classes."""
    all_feats = []
    for cls in class_names:
        all_feats.extend(importance_dict[cls]['feature'].tolist())
    top_feats = list(dict.fromkeys(all_feats))[:12]  # preserve order, deduplicate

    data = {}
    for cls in class_names:
        df = importance_dict[cls].set_index('feature')
        data[cls] = [df.loc[f, 'mean_abs_shap'] if f in df.index else 0.0
                     for f in top_feats]

    x   = np.arange(len(top_feats))
    w   = 0.25
    colors = list(PALETTE.values())
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        ax.bar(x + i*w, data[cls], w, label=cls, color=color, alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(top_feats, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP|')
    ax.set_title('Linguistic Feature Importance by Class (SHAP)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_shap_heatmap(explainer: shap.LinearExplainer,
                       X_sample,
                       feature_names: List[str],
                       class_names: List[str],
                       class_idx: int = 0,
                       n_samples: int = 30,
                       top_n_features: int = 15,
                       save_path: Optional[str] = None) -> None:
    """Heatmap: samples (rows) × top features (cols), coloured by SHAP value."""
    shap_arr = np.array(explainer.shap_values(X_sample[:n_samples]))
    if shap_arr.ndim == 2:
        sv = shap_arr
    else:
        sv = shap_arr[:, :, class_idx]

    top_idx   = np.abs(sv).mean(0).argsort()[-top_n_features:][::-1]
    sv_top    = sv[:, top_idx]
    feat_top  = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, max(5, n_samples * 0.22)))
    sns.heatmap(sv_top, xticklabels=feat_top, yticklabels=False,
                cmap='RdBu_r', center=0, ax=ax, linewidths=0,
                cbar_kws={'label': 'SHAP value', 'shrink': 0.6})
    ax.set_xlabel('Feature')
    ax.set_ylabel(f'Sample (n={n_samples})')
    ax.set_title(f'SHAP Heatmap — Class: {class_names[class_idx]}',
                 fontweight='bold')
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


# ── Inference-time token highlights (used by Flask app) ────────────────────────

def get_token_highlights(text,
                         explainer,
                         vectorizer,
                         top_n=5):
    """Return top token SHAP contributions for a raw text."""

    X = vectorizer.transform([text])

    sv_arr = np.array(explainer.shap_values(X))

    if sv_arr.ndim == 2:
        sv_pred = sv_arr[0]
    else:
        class_sums = np.abs(sv_arr[0]).sum(axis=0)
        pred_cls = int(np.argmax(class_sums))
        sv_pred = sv_arr[0, :, pred_cls]

    feat_names = vectorizer.get_feature_names_out()
    top_idx = np.abs(sv_pred).argsort()[-top_n:][::-1]

    highlights = []
    for i in top_idx:
        highlights.append({
            "token": feat_names[i],
            "shap": float(sv_pred[i]),
            "direction": "positive" if sv_pred[i] > 0 else "negative"
        })

    return highlights

if __name__ == '__main__':
    print("shap_utils.py loaded successfully.")
    print("Exports: build_tfidf_explainer, build_linguistic_explainer,")
    print("         global_token_importance, global_linguistic_importance,")
    print("         individual_shap_breakdown, plot_global_token_importance,")
    print("         plot_waterfall, plot_linguistic_feature_importance,")
    print("         plot_shap_heatmap, get_token_highlights")
