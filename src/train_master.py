"""
train_master.py - MASTER paneli için tam eğitim scripti
  • Balanced bagging: tüm benign + 2× patojenik alt-örneklem, 3 bag/fold
  • Group-aware CV: StratifiedGroupKFold — aynı kayıtlar aynı fold
  • Ensemble: CatBoost(×2) + LightGBM(×1) + XGBoost(×1) ağırlıklı soft voting
  • Eşik: MCC + benign recall + patojenik recall kombinasyonu (OOF, 901 adım)
  • NaN temizliği: tamamen boş ve %80+ NaN sütunlar düşürülür, medyan doldurulur

Kullanım:
    python src/train_master.py
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, classification_report,
    matthews_corrcoef, recall_score, balanced_accuracy_score,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

PANEL        = "MASTER"
ROOT         = Path(__file__).parent.parent
RAW_DIR      = ROOT / "data" / "raw"
OUTPUTS_DIR  = ROOT / "outputs"
METRICS_DIR  = OUTPUTS_DIR / "metrics"
TARGET       = "Label"
N_FOLDS      = 5
RANDOM_STATE = 42
TOP_N        = 20

BALANCED_BAGS              = 3
PATHOGENIC_TO_BENIGN_RATIO = 2.0
MIN_PATHOGENIC_RECALL      = 0.90

CAT_PARAMS = {
    'iterations':         600,
    'learning_rate':      0.03,
    'depth':              4,
    'l2_leaf_reg':        5,
    'loss_function':      'Logloss',
    'eval_metric':        'AUC',
    'auto_class_weights': 'Balanced',
    'random_seed':        60,
    'verbose':            0,
    'allow_writing_files': False,
    'thread_count':       -1,
}

LGBM_PARAMS = {
    'n_estimators':      400,
    'learning_rate':     0.04,
    'num_leaves':        15,
    'max_depth':         4,
    'min_child_samples': 12,
    'subsample':         0.80,
    'subsample_freq':    1,
    'colsample_bytree':  0.80,
    'reg_alpha':         0.10,
    'reg_lambda':        1.00,
    'objective':         'binary',
    'random_state':      RANDOM_STATE,
    'n_jobs':            -1,
    'verbosity':         -1,
}

XGB_PARAMS = {
    'n_estimators':     400,
    'learning_rate':    0.04,
    'max_depth':        4,
    'min_child_weight': 5.0,
    'subsample':        0.80,
    'colsample_bytree': 0.80,
    'gamma':            1.0,
    'reg_alpha':        0.10,
    'reg_lambda':       2.00,
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'tree_method':      'hist',
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
    'verbosity':        0,
}

ENSEMBLE_WEIGHTS = [2, 1, 1]  # CatBoost, LightGBM, XGBoost


def sep(title=""):
    print(f"\n{'=' * 62}")
    if title:
        print(f"  {title}")
        print("=" * 62)


def load_features():
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    raw_p = pd.read_csv(RAW_DIR / "YARISMA_TRAIN_MASTER_2.csv", low_memory=False)
    df = raw_p.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")

    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes

    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

    df = add_bio_features(df, raw_p)

    # NaN temizliği
    n_rows_before = len(df); n_cols_before = df.shape[1]
    df = df.dropna(axis=1, how='all')
    label_col = df.pop(TARGET) if TARGET in df.columns else None
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2))
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    if label_col is not None:
        df[TARGET] = label_col.values
    print(f"  [NaN] {n_cols_before} sutun -> {df.shape[1]} sutun | "
          f"{n_rows_before} satir -> {len(df)} satir")

    return df, raw_p


def make_exact_row_groups(raw_x: pd.DataFrame) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(raw_x, index=False)
    return pd.factorize(hashes, sort=False)[0]


def inverse_duplicate_weights(groups: np.ndarray) -> np.ndarray:
    group_series = pd.Series(groups)
    group_sizes = group_series.map(group_series.value_counts()).to_numpy()
    return 1.0 / group_sizes


def make_balanced_bag_indices(y_train: np.ndarray, bag_num: int, fold_num: int) -> np.ndarray:
    benign_idx = np.flatnonzero(y_train == 0)
    pathogenic_idx = np.flatnonzero(y_train == 1)
    sample_size = min(
        int(np.ceil(len(benign_idx) * PATHOGENIC_TO_BENIGN_RATIO)),
        len(pathogenic_idx),
    )
    rng = np.random.default_rng(RANDOM_STATE + 1000 * fold_num + 37 * bag_num)
    sampled = rng.choice(pathogenic_idx, size=sample_size, replace=False)
    selected = np.concatenate([benign_idx, sampled])
    rng.shuffle(selected)
    return selected


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    best, fallback = None, None
    for t in np.linspace(0.05, 0.95, 901):
        preds  = (y_prob >= t).astype(int)
        path_r = recall_score(y_true, preds, pos_label=1, zero_division=0)
        ben_r  = recall_score(y_true, preds, pos_label=0, zero_division=0)
        mcc    = matthews_corrcoef(y_true, preds)
        bal    = balanced_accuracy_score(y_true, preds)
        score  = 0.55 * mcc + 0.25 * ben_r + 0.10 * path_r + 0.10 * bal
        entry  = (float(t), float(score))
        if path_r >= MIN_PATHOGENIC_RECALL:
            if best is None or score > best[1]:
                best = entry
        if fallback is None or score > fallback[1]:
            fallback = entry
    return best if best is not None else fallback


def build_lgbm(n_neg: int, n_pos: int) -> LGBMClassifier:
    p = dict(LGBM_PARAMS)
    p['scale_pos_weight'] = n_neg / max(n_pos, 1)
    return LGBMClassifier(**p)


def build_xgb(n_neg: int, n_pos: int) -> XGBClassifier:
    p = dict(XGB_PARAMS)
    p['scale_pos_weight'] = n_neg / max(n_pos, 1)
    return XGBClassifier(**p)


def ensemble_prob(models: list, X: pd.DataFrame) -> np.ndarray:
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return np.average(probs, axis=1, weights=ENSEMBLE_WEIGHTS)


def print_importance_table(imp: pd.Series, top_n: int = TOP_N):
    top = imp.head(top_n).reset_index()
    top.columns = ["Ozellik", "Onem"]
    top["Onem%"] = (top["Onem"] / imp.sum() * 100).round(2)
    top["Grup"] = top["Ozellik"].apply(
        lambda c: "BIO-BLOK" if "alblk" in c else
                  "BIO"      if c.startswith("BIO_") else
                  "EK"       if c.startswith("EK_")  else
                  "AL"       if c.startswith("AL_")  else
                  "MISSING"  if c.startswith("MISSING_") else "DIGER"
    )
    top.index = range(1, len(top) + 1)
    print(f"\n  {'#':>3}  {'Ozellik':<26}  {'Onem%':>6}  {'Grup':<8}  Gorsel")
    print(f"  {'-'*3}  {'-'*26}  {'-'*6}  {'-'*8}  {'-'*22}")
    for i, row in top.iterrows():
        bar = "|" * max(1, int(row["Onem%"] / 2))
        print(f"  {i:>3}  {row['Ozellik']:<26}  {row['Onem%']:>5.1f}%  {row['Grup']:<8}  {bar}")
    grp_totals = top.groupby("Grup")["Onem%"].sum().sort_values(ascending=False)
    print()
    print(f"  {'Grup':<10}  {'Toplam%':>7}  Gorsel")
    print(f"  {'-'*10}  {'-'*7}  {'-'*20}")
    for grp, pct in grp_totals.items():
        bar = "#" * int(pct / 3)
        print(f"  {grp:<10}  {pct:>6.1f}%  {bar}")


def train():
    sep(f"PANEL: {PANEL} — Balanced Bagging + Group-Aware CV")

    feat_df, raw_p = load_features()
    X = feat_df.drop(columns=[TARGET], errors="ignore")
    y = feat_df[TARGET]

    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    print(f"  Satirlar   : {len(y)}")
    print(f"  POS (1)    : {n_pos}   NEG (0): {n_neg}   Oran: {y.mean():.3f}")
    print(f"  Ozellikler : {X.shape[1]}")

    if n_neg < 5:
        print(f"  [UYARI] Cok az negatif ornek ({n_neg}), egitim iptal.")
        return

    # ── Gruplama ──────────────────────────────────────────────
    raw_feat    = raw_p.drop(columns=["Variant_ID", TARGET, "panel"], errors="ignore")
    groups      = make_exact_row_groups(raw_feat)
    dup_weights = inverse_duplicate_weights(groups)

    gc = pd.Series(groups).value_counts()
    print(f"  Dup. grup  : {int((gc > 1).sum())} grup / {int(gc[gc > 1].sum())} satir")

    # ── Group-Aware OOF ───────────────────────────────────────
    n_splits = min(N_FOLDS, n_neg)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_prob  = np.zeros(len(y))
    fold_imps = []
    fold_rows = []

    sep(f"{n_splits}-FOLD GROUP-AWARE OOF  "
        f"[CatBoost×2 + LightGBM×1 + XGBoost×1, {BALANCED_BAGS} bag]")
    print(f"  {'Fold':>4}  {'AUC':>6}  {'F1-NEG':>6}  {'F1-POS':>6}  Top-5 ozellik")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*40}")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        fold_dw    = dup_weights[tr_idx]

        cat_probs, lgb_probs, xgb_probs = [], [], []

        for bag in range(1, BALANCED_BAGS + 1):
            bag_idx = make_balanced_bag_indices(y_tr.values, bag, fold)
            X_bag   = X_tr.iloc[bag_idx]
            y_bag   = y_tr.iloc[bag_idx]
            w_bag   = fold_dw[bag_idx]

            n_b_neg = int((y_bag == 0).sum())
            n_b_pos = int((y_bag == 1).sum())

            cb  = CatBoostClassifier(**CAT_PARAMS)
            lgb = build_lgbm(n_b_neg, n_b_pos)
            xgb = build_xgb(n_b_neg, n_b_pos)

            cb.fit(X_bag, y_bag, sample_weight=w_bag)
            lgb.fit(X_bag, y_bag, sample_weight=w_bag)
            xgb.fit(X_bag, y_bag, sample_weight=w_bag)

            cat_probs.append(cb.predict_proba(X_te)[:, 1])
            lgb_probs.append(lgb.predict_proba(X_te)[:, 1])
            xgb_probs.append(xgb.predict_proba(X_te)[:, 1])

            if bag == BALANCED_BAGS:
                fold_imps.append(
                    pd.Series(cb.feature_importances_, index=X.columns)
                    .sort_values(ascending=False)
                )

        fold_prob = np.average(
            np.column_stack([
                np.mean(cat_probs, axis=0),
                np.mean(lgb_probs, axis=0),
                np.mean(xgb_probs, axis=0),
            ]),
            axis=1,
            weights=ENSEMBLE_WEIGHTS,
        )
        oof_prob[te_idx] = fold_prob

        try:
            auc = roc_auc_score(y_te, fold_prob)
        except ValueError:
            auc = float("nan")
        pred_05 = (fold_prob >= 0.5).astype(int)
        f1n = f1_score(y_te, pred_05, pos_label=0, zero_division=0)
        f1p = f1_score(y_te, pred_05, pos_label=1, zero_division=0)

        imp  = fold_imps[-1]
        top5 = ", ".join(f"{c}({v:.1f}%)"
                         for c, v in zip(imp.head(5).index,
                                         imp.head(5).values / imp.sum() * 100))
        print(f"  {fold:>4}  {auc:>6.4f}  {f1n:>6.4f}  {f1p:>6.4f}  {top5}")
        fold_rows.append({"fold": fold, "auc": auc, "f1_neg_05": f1n, "f1_pos_05": f1p})

    # ── Dinamik eşik ──────────────────────────────────────────
    sep("ESIK BELIRLEME  (MCC + Benign Recall + Path. Recall, 901 adim)")
    threshold, _ = find_best_threshold(y.values, oof_prob)

    t05_mf1    = f1_score(y.values, (oof_prob >= 0.50).astype(int),
                          average="macro", zero_division=0)
    thresh_mf1 = f1_score(y.values, (oof_prob >= threshold).astype(int),
                          average="macro", zero_division=0)

    print(f"\n  Bulunan esik  : {threshold:.4f}")
    print(f"  Macro F1 (OOF): {thresh_mf1:.4f}")
    print(f"\n  Karsilastirma:")
    print(f"    Sabit 0.50   -> Macro F1 = {t05_mf1:.4f}")
    print(f"    Dinamik {threshold:.4f} -> Macro F1 = {thresh_mf1:.4f}"
          f"  (+{thresh_mf1 - t05_mf1:.4f})")

    # ── OOF Metrikleri ────────────────────────────────────────
    all_pred = (oof_prob >= threshold).astype(int)
    oof_auc  = roc_auc_score(y.values, oof_prob)
    oof_mcc  = matthews_corrcoef(y.values, all_pred)
    oof_mf1  = f1_score(y.values, all_pred, average="macro", zero_division=0)
    oof_f1n  = f1_score(y.values, all_pred, pos_label=0, zero_division=0)
    oof_f1p  = f1_score(y.values, all_pred, pos_label=1, zero_division=0)
    path_r   = recall_score(y.values, all_pred, pos_label=1, zero_division=0)
    ben_r    = recall_score(y.values, all_pred, pos_label=0, zero_division=0)

    sep("OOF TEST METRIKLERI  (Ensemble + Dinamik Esik)")
    print(f"\n  {'Metrik':<24}  {'Deger':>8}")
    print(f"  {'-'*24}  {'-'*8}")
    print(f"  {'ROC-AUC':<24}  {oof_auc:>8.4f}")
    print(f"  {'MCC':<24}  {oof_mcc:>8.4f}")
    print(f"  {'Macro F1':<24}  {oof_mf1:>8.4f}  <-- hedef")
    print(f"  {'F1 Benign (NEG)':<24}  {oof_f1n:>8.4f}")
    print(f"  {'F1 Patojenik (POS)':<24}  {oof_f1p:>8.4f}")
    print(f"  {'Patojenik Recall':<24}  {path_r:>8.4f}")
    print(f"  {'Benign Recall':<24}  {ben_r:>8.4f}")
    print(f"  {'Esik':<24}  {threshold:>8.4f}")

    cm = confusion_matrix(y.values, all_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:  [esik={threshold:.4f}]")
    print(f"  {'':20s}  Pred-NEG  Pred-POS")
    print(f"  {'Gercek Benign (0)':<20}  {tn:>8d}  {fp:>8d}  (n={n_neg})")
    print(f"  {'Gercek Patojenik (1)':<20}  {fn:>8d}  {tp:>8d}  (n={n_pos})")
    print(f"\n  Classification Report:  [esik={threshold:.4f}]")
    print(classification_report(y.values, all_pred,
                                target_names=["Benign (0)", "Patojenik (1)"],
                                zero_division=0))

    # ── Özellik önemi ─────────────────────────────────────────
    avg_imp = pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)
    sep(f"OZELLIK ONEMLERI — TOP {TOP_N}  (CatBoost fold ortalaması)")
    print_importance_table(avg_imp, top_n=TOP_N)

    # ── Final model — tüm veri ────────────────────────────────
    sep("FINAL MODEL  (tum veri ile egitim)")
    final_cb  = CatBoostClassifier(**CAT_PARAMS)
    final_lgb = build_lgbm(n_neg, n_pos)
    final_xgb = build_xgb(n_neg, n_pos)
    final_cb.fit(X, y, sample_weight=dup_weights)
    final_lgb.fit(X, y, sample_weight=dup_weights)
    final_xgb.fit(X, y, sample_weight=dup_weights)
    print("  3 model egitildi (CatBoost, LightGBM, XGBoost).")

    # ── Kaydet ───────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    model_path = OUTPUTS_DIR / f"model_{PANEL}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models":           [final_cb, final_lgb, final_xgb],
            "threshold":        threshold,
            "features":         list(X.columns),
            "panel":            PANEL,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
        }, f)

    result = {
        "panel":             PANEL,
        "threshold":         round(threshold, 4),
        "oof_auc":           round(float(oof_auc), 4),
        "oof_mcc":           round(float(oof_mcc), 4),
        "macro_f1":          round(oof_mf1, 4),
        "f1_neg":            round(oof_f1n, 4),
        "f1_pos":            round(oof_f1p, 4),
        "pathogenic_recall": round(float(path_r), 4),
        "benign_recall":     round(float(ben_r), 4),
    }
    with open(METRICS_DIR / f"panel_{PANEL}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    avg_imp.to_csv(METRICS_DIR / f"panel_{PANEL}_importance.csv", header=["importance"])
    pd.DataFrame(fold_rows).to_csv(METRICS_DIR / f"panel_{PANEL}_folds.csv", index=False)

    print(f"  Model     : {model_path}")
    print(f"  Metrikler : {METRICS_DIR / f'panel_{PANEL}_metrics.json'}")

    print(f"\n{'=' * 62}")
    print(f"  SONUC: {PANEL}  AUC={oof_auc:.4f}  MCC={oof_mcc:.4f}  "
          f"MacroF1={oof_mf1:.4f}  Esik={threshold:.4f}")
    print(f"  PathRecall={path_r:.4f}  BenignRecall={ben_r:.4f}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    train()
