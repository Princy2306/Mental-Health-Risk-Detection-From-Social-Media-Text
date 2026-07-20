"""
train.py — Clean, reproducible training script for mental health NLP baselines.

Usage:
    python train.py                        # train all classical models
    python train.py --model lr             # train LogisticRegression only
    python train.py --model svm            # train LinearSVC only
    python train.py --model rf             # train RandomForest only
    python train.py --features combined    # use TF-IDF + linguistic features
    python train.py --features tfidf       # use TF-IDF only (default)
    python train.py --tune                 # run GridSearchCV hyperparameter tuning
    python train.py --smote                # apply SMOTE before training

Example (full run):
    python train.py --features combined --tune --smote
"""

import os, sys, argparse, time, warnings, json
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                      train_test_split)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from features  import extract_features_batch, clean_texts_batch
from evaluate  import (compute_metrics, cross_val_metrics,
                        measure_inference_time, plot_confusion_matrix,
                        plot_cv_comparison)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(ROOT, 'models')
DOCS_DIR  = os.path.join(ROOT, 'docs')
RESULTS_F = os.path.join(DATA_DIR, 'results_table.csv')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOCS_DIR,  exist_ok=True)

LABEL_ORDER = ['low', 'moderate', 'high']
RANDOM_STATE = 42


# ── Hyperparameter grids ───────────────────────────────────────────────────────
PARAM_GRIDS = {
    'lr': {
        'C'      : [0.01, 0.1, 1.0, 5.0, 10.0],
        'solver' : ['lbfgs', 'liblinear'],
        'penalty': ['l2'],
    },
    'svm': {
        'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    },
    'rf': {
        'n_estimators': [100, 200],
        'max_depth'   : [None, 20, 40],
        'min_samples_split': [2, 5],
    },
}

# ── Model definitions ──────────────────────────────────────────────────────────
def get_model(name: str):
    return {
        'lr' : LogisticRegression(
                    class_weight='balanced', max_iter=1000,
                    random_state=RANDOM_STATE, C=1.0),
        'svm': LinearSVC(
                    class_weight='balanced', max_iter=2000,
                    random_state=RANDOM_STATE, C=1.0),
        'rf' : RandomForestClassifier(
                    n_estimators=200, class_weight='balanced',
                    random_state=RANDOM_STATE, n_jobs=-1),
    }[name]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(feature_set: str = 'tfidf'):
    """Load features and labels. Returns X (sparse or dense), y, texts, le."""
    # Prefer combined dataset if available
    combined_path = os.path.join(DATA_DIR, 'features_combined.csv')
    kaggle_path   = os.path.join(DATA_DIR, 'features_kaggle.csv')
    data_path     = combined_path if os.path.exists(combined_path) else kaggle_path
    print(f"[data] Loading: {os.path.basename(data_path)}")

    feat_df = pd.read_csv(data_path)
    le      = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    vec     = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

    texts       = feat_df['text_clean'].fillna('').tolist()
    y           = le.transform(feat_df['label'])
    X_tfidf     = vec.transform(texts)

    if feature_set == 'combined':
        ling_cols = [c for c in feat_df.columns
                     if c not in ['label', 'label_enc', 'text_clean', 'source']]
        X_ling    = feat_df[ling_cols].fillna(0).values
        X         = sp.hstack([X_tfidf, sp.csr_matrix(X_ling)])
        print(f"[data] Combined features: TF-IDF({X_tfidf.shape[1]}) + "
              f"linguistic({X_ling.shape[1]}) = {X.shape[1]} total")
    else:
        X = X_tfidf
        print(f"[data] TF-IDF features: {X.shape}")

    print(f"[data] Labels: {dict(zip(le.classes_, np.bincount(y)))}")
    return X, y, texts, le


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(name: str, X, y, le, tune: bool = False,
                apply_smote: bool = False) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training: {name.upper()}")
    print(f"{'='*55}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # SMOTE on training set only
    if apply_smote:
        print("[smote] Applying SMOTE to training set...")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        print(f"[smote] After: {np.bincount(y_tr)}")

    model = get_model(name)

    # Hyperparameter tuning
    if tune:
        print(f"[tune] GridSearchCV (5-fold, scoring=f1_macro)...")
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(model, PARAM_GRIDS[name], cv=cv_inner,
                          scoring='f1_macro', n_jobs=-1, verbose=0)
        t0 = time.time()
        gs.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        model = gs.best_estimator_
        print(f"[tune] Best params : {gs.best_params_}")
        print(f"[tune] Best CV F1  : {gs.best_score_:.4f}  ({elapsed:.1f}s)")
    else:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        print(f"[train] Fit time: {time.time()-t0:.2f}s")

    # Evaluation
    y_pred = model.predict(X_te)
    test_metrics = compute_metrics(y_te, y_pred, le.classes_.tolist())
    cv_metrics   = cross_val_metrics(model, X, y, cv=5)
    infer_ms     = measure_inference_time(model, X_te)

    # Model size
    tmp_path = os.path.join(MODEL_DIR, f'_tmp_{name}.pkl')
    joblib.dump(model, tmp_path, compress=3)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.rename(tmp_path, os.path.join(MODEL_DIR, f'{name}_model.pkl'))

    print(f"\n[results] Test set metrics:")
    print(f"  Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"  F1-macro          : {test_metrics['f1_macro']:.4f}")
    print(f"  F1-weighted       : {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision-macro   : {test_metrics['precision_macro']:.4f}")
    print(f"  Recall-macro      : {test_metrics['recall_macro']:.4f}")
    print(f"  Per-class F1      : {test_metrics['per_class_f1']}")
    print(f"\n[results] CV metrics (5-fold):")
    print(f"  CV F1-macro       : {cv_metrics['cv_f1_macro']:.4f} "
          f"± {cv_metrics['cv_f1_macro_std']:.4f}")
    print(f"  Overfit gap       : {cv_metrics['overfit_gap']:.4f}")
    print(f"\n[results] Efficiency:")
    print(f"  Inference         : {infer_ms:.2f} ms/sample")
    print(f"  Model size        : {size_kb:.1f} KB")

    # Confusion matrix
    plot_confusion_matrix(
        y_te, y_pred, le.classes_.tolist(),
        title=f'Confusion Matrix — {name.upper()}',
        save_path=os.path.join(DOCS_DIR, f'plot_cm_{name}.png')
    )

    return {
        'model_name'    : name,
        'feature_set'   : 'combined' if X.shape[1] > 10001 else 'tfidf',
        'smote'         : apply_smote,
        'tuned'         : tune,
        **test_metrics,
        **cv_metrics,
        'inference_ms'  : infer_ms,
        'model_size_kb' : round(size_kb, 1),
        'model_obj'     : model,
        'y_te'          : y_te,
        'y_pred'        : y_pred,
    }


# ── Results table ──────────────────────────────────────────────────────────────
def save_results(all_results: list) -> pd.DataFrame:
    """Append results to the running results CSV."""
    cols = ['model_name','feature_set','smote','tuned','accuracy',
            'f1_macro','f1_weighted','precision_macro','recall_macro',
            'cv_f1_macro','cv_f1_macro_std','overfit_gap',
            'inference_ms','model_size_kb']
    rows = [{c: r[c] for c in cols} for r in all_results]
    df_new = pd.DataFrame(rows)

    if os.path.exists(RESULTS_F):
        df_old = pd.read_csv(RESULTS_F)
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            subset=['model_name','feature_set','smote','tuned'], keep='last')
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_F, index=False)
    print(f"\n[saved] Results table → {RESULTS_F}")
    print(df_all[['model_name','feature_set','f1_macro','cv_f1_macro',
                   'inference_ms','model_size_kb']].to_string(index=False))
    return df_all


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Train mental health NLP baselines')
    p.add_argument('--model',    choices=['lr','svm','rf','all'], default='all')
    p.add_argument('--features', choices=['tfidf','combined'],    default='tfidf')
    p.add_argument('--tune',     action='store_true')
    p.add_argument('--smote',    action='store_true')
    return p.parse_args()


def main():
    args  = parse_args()
    X, y, texts, le = load_data(args.features)

    model_names = ['lr','svm','rf'] if args.model == 'all' else [args.model]
    all_results = []

    for name in model_names:
        result = train_model(name, X, y, le,
                             tune=args.tune, apply_smote=args.smote)
        all_results.append(result)

    results_df = save_results(all_results)

    if len(all_results) > 1:
        cv_dict = {r['model_name']: {
            'cv_f1_macro'    : r['cv_f1_macro'],
            'cv_f1_macro_std': r['cv_f1_macro_std'],
            'overfit_gap'    : r['overfit_gap'],
        } for r in all_results}
        plot_cv_comparison(cv_dict,
            save_path=os.path.join(DOCS_DIR, 'plot_cv_comparison.png'))

    print("\n[done] Week 3 training complete.")
    print(f"  Best model by F1-macro: "
          f"{results_df.dropna(subset=["model_name"]).sort_values('f1_macro',ascending=False).iloc[0]['model_name'].upper()}")


if __name__ == '__main__':
    main()
