"""
fairness.py — Demographic fairness audit for mental health NLP project.

Computes group-wise metrics and flags disparities following
AI Fairness 360 conventions (demographic parity difference, equalised odds).

Used by: notebooks/04_shap_fairness.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Optional
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Thresholds (based on AI Fairness 360 + clinical ML literature)
DPD_WARN  = 0.05   # Demographic Parity Difference — flag if exceeded
DPD_ALERT = 0.10   # Raise stronger concern
EOD_WARN  = 0.05   # Equalized Odds Difference — flag if exceeded

PALETTE = {'high': '#D85A30', 'moderate': '#BA7517', 'low': '#1D9E75'}


# ── Per-group metrics ──────────────────────────────────────────────────────────

def compute_group_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           groups: np.ndarray,
                           class_names: List[str],
                           high_risk_label: str = 'high',
                           min_group_size: int = 10) -> pd.DataFrame:
    """
    Compute F1-macro, recall, precision, and accuracy per demographic group.

    Parameters
    ----------
    y_true, y_pred : encoded label arrays (integers)
    groups         : array of group labels (e.g. gender strings) — same length as y_true
    class_names    : le.classes_ list
    high_risk_label: which class label corresponds to 'high risk' (for recall audit)
    min_group_size : skip groups smaller than this (unreliable estimates)

    Returns DataFrame with one row per group and metric columns.
    """
    high_risk_idx = list(class_names).index(high_risk_label) \
                    if high_risk_label in class_names else None

    rows = []
    for grp in np.unique(groups):
        mask = groups == grp
        if mask.sum() < min_group_size:
            print(f"[fairness] Skipping group '{grp}' — only {mask.sum()} samples (< {min_group_size})")
            continue

        yt, yp = y_true[mask], y_pred[mask]
        row = {
            'group'           : grp,
            'n'               : int(mask.sum()),
            'accuracy'        : round(accuracy_score(yt, yp), 4),
            'f1_macro'        : round(f1_score(yt, yp, average='macro',    zero_division=0), 4),
            'f1_weighted'     : round(f1_score(yt, yp, average='weighted', zero_division=0), 4),
            'precision_macro' : round(precision_score(yt, yp, average='macro', zero_division=0), 4),
            'recall_macro'    : round(recall_score(yt, yp, average='macro', zero_division=0), 4),
        }
        # Per-class F1 (flat columns)
        per_class_f1 = f1_score(yt, yp, average=None, zero_division=0, labels=list(range(len(class_names))))
        for i, cls in enumerate(class_names):
            row[f'f1_{cls}'] = round(per_class_f1[i], 4)

        # High-risk recall (clinical priority metric)
        if high_risk_idx is not None:
            hr_recall = recall_score(yt, yp, labels=[high_risk_idx],
                                     average='macro', zero_division=0)
            row['recall_high_risk'] = round(hr_recall, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Disparity measures ─────────────────────────────────────────────────────────

def demographic_parity_difference(group_df: pd.DataFrame,
                                   metric: str = 'f1_macro') -> Dict:
    """
    Demographic Parity Difference (DPD) = max(metric) - min(metric) across groups.
    A DPD of 0 = perfect parity. >0.05 = flag. >0.10 = significant disparity.
    """
    vals    = group_df[metric].values
    best_g  = group_df.loc[group_df[metric].idxmax(), 'group']
    worst_g = group_df.loc[group_df[metric].idxmin(), 'group']
    dpd     = float(vals.max() - vals.min())

    severity = 'OK' if dpd < DPD_WARN else ('WARN' if dpd < DPD_ALERT else 'ALERT')
    return {
        'metric'          : metric,
        'dpd'             : round(dpd, 4),
        'severity'        : severity,
        'best_group'      : best_g,
        'worst_group'     : worst_g,
        'best_score'      : round(vals.max(), 4),
        'worst_score'     : round(vals.min(), 4),
        'recommendation'  : {
            'OK'   : 'No action required. Monitor on larger datasets.',
            'WARN' : f'Investigate gap between {best_g} and {worst_g}. '
                     f'Consider group-stratified resampling or re-weighting.',
            'ALERT': f'Significant disparity detected ({best_g} vs {worst_g}, Δ={dpd:.3f}). '
                     f'Do NOT deploy without mitigation. Options: re-collect balanced data, '
                     f'use group-aware loss, post-process predictions.',
        }[severity],
    }


def print_fairness_report(group_df: pd.DataFrame,
                          class_names: List[str]) -> None:
    """Print fairness metrics."""

    print(f"\nGroup-level performance ({len(group_df)} groups):")

    display_cols = ['group', 'n', 'f1_macro', 'recall_macro'] + \
                   [f'f1_{c}' for c in class_names if f'f1_{c}' in group_df.columns] + \
                   (['recall_high_risk'] if 'recall_high_risk' in group_df.columns else [])

    print(group_df[display_cols].to_string(index=False))

    print("\nDisparity measures:")

    for metric in ['f1_macro', 'recall_macro', 'recall_high_risk']:
        if metric not in group_df.columns:
            continue

        dpd_res = demographic_parity_difference(group_df, metric)

        print(f"DPD ({metric}): {dpd_res['dpd']:.4f}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_fairness_bars(group_df: pd.DataFrame,
                        class_names: List[str],
                        save_path: Optional[str] = None) -> None:
    """Side-by-side bars: F1-macro and high-risk recall per demographic group."""
    metrics = ['f1_macro', 'recall_macro']
    if 'recall_high_risk' in group_df.columns:
        metrics.append('recall_high_risk')

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    group_colors = plt.cm.Set2(np.linspace(0, 1, len(group_df)))

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(group_df['group'], group_df[metric],
                      color=group_colors, alpha=0.88, width=0.55)
        # Threshold line
        ax.axhline(group_df[metric].mean(), color='black',
                   linestyle='--', linewidth=0.9, label='Mean')
        for bar, val in zip(bars, group_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=9, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # DPD annotation
        dpd_res = demographic_parity_difference(group_df, metric)
        color   = {'OK':'#1D9E75','WARN':'#BA7517','ALERT':'#D85A30'}[dpd_res['severity']]
        ax.text(0.98, 0.03, f"DPD={dpd_res['dpd']:.3f} [{dpd_res['severity']}]",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color=color, fontweight='bold')

    plt.suptitle('Fairness Audit — Group-Level Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_per_class_fairness(group_df: pd.DataFrame,
                              class_names: List[str],
                              save_path: Optional[str] = None) -> None:
    """Heatmap: groups × classes, cell = F1 score. Reveals class-specific disparities."""
    cls_cols = [f'f1_{c}' for c in class_names if f'f1_{c}' in group_df.columns]
    if not cls_cols:
        print("[fairness] No per-class F1 columns found. Skipping heatmap.")
        return

    matrix = group_df.set_index('group')[cls_cols].rename(
        columns={f'f1_{c}': c for c in class_names})

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 2.5),
                                     max(4, len(group_df) * 1.0)))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'F1 score', 'shrink': 0.7},
                annot_kws={'size': 11})
    ax.set_title('Per-Class F1 by Demographic Group', fontweight='bold', fontsize=12)
    ax.set_xlabel('Class')
    ax.set_ylabel('Group')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


def plot_prediction_distribution(y_pred: np.ndarray,
                                  groups: np.ndarray,
                                  class_names: List[str],
                                  save_path: Optional[str] = None) -> None:
    """
    Stacked bar: predicted class distribution per group.
    A perfectly fair model would show the same distribution across groups.
    """
    unique_groups = np.unique(groups)
    data = {}
    for grp in unique_groups:
        mask = groups == grp
        counts = np.bincount(y_pred[mask], minlength=len(class_names))
        data[grp] = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(max(6, len(unique_groups) * 1.5), 5))
    bottom = np.zeros(len(unique_groups))
    colors = ['#1D9E75', '#BA7517', '#D85A30', '#7F77DD'][:len(class_names)]

    for i, cls in enumerate(class_names):
        vals = [data[g][i] for g in unique_groups]
        ax.bar(unique_groups, vals, bottom=bottom,
               label=cls, color=colors[i], alpha=0.85)
        bottom += np.array(vals)

    ax.set_ylabel('Proportion of predictions')
    ax.set_title('Predicted Class Distribution by Group\n(Parity = same distribution across groups)',
                 fontweight='bold')
    ax.legend(title='Predicted class', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=130)
    plt.show()


if __name__ == '__main__':
    print("fairness.py loaded successfully.")
    print("Exports: compute_group_metrics, demographic_parity_difference,")
    print("         print_fairness_report, plot_fairness_bars,")
    print("         plot_per_class_fairness, plot_prediction_distribution")
