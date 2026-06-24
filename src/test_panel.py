"""
test_panel.py - Kayıtlı modeli kullanarak hızlı OOF değerlendirme.
Optuna yok - best_params ve threshold doğrudan model.pkl'den alınır.

Çıktı: MCC, ROC-AUC, ROC eğrisi tablosu, Confusion Matrix,
       Classification Report, Feature Importance tablosu.

Kullanım:
    python src/test_panel.py CFTR
    python src/test_panel.py KANSER
"""
import sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, f1_score,
    confusion_matrix, classification_report, roc_curve,
)

warnings.filterwarnings("ignore")
ROOT        = Path(__file__).parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
TARGET        = "Label"
RANDOM_STATE  = 42


def sep(title=""):
    print(f"\n{'=' * 62}")
    if title:
        print(f"  {title}")
        print("=" * 62)


def load_panel_features(panel: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    raw_p = pd.read_csv(RAW_DIR / f"YARISMA_TRAIN_{panel}.csv", low_memory=False)
    df    = raw_p.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")

    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes

    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)
    df = add_bio_features(df, raw_p)
    return df


def find_best_threshold(y_true, y_prob, lo=0.20, hi=0.80, steps=121):
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(lo, hi, steps):
        score = f1_score(y_true, (y_prob >= t).astype(int),
                         average="macro", zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def ensemble_prob(models, X):
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return probs.mean(axis=1)


def print_roc_table(fpr, tpr, thresholds, n_points=12):
    idxs = np.linspace(0, len(fpr) - 1, n_points, dtype=int)
    print(f"\n  {'Esik':>7}  {'FPR':>6}  {'TPR/Sens':>8}  {'Ozg(1-FPR)':>10}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*10}")
    for i in idxs:
        th   = thresholds[min(i, len(thresholds) - 1)]
        spec = 1.0 - fpr[i]
        print(f"  {th:>7.4f}  {fpr[i]:>6.3f}  {tpr[i]:>8.3f}  {spec:>10.3f}")


def print_importance_table(imp: pd.Series, top_n: int = 20):
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


def test_panel(panel: str):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    model_path = OUTPUTS_DIR / f"model_{panel}.pkl"
    if not model_path.exists():
        print(f"\n  [HATA] {model_path} bulunamadi.")
        print(f"  Once: python src/train_{panel.lower()}.py")
        sys.exit(1)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    final_models = saved["models"]
    features     = saved["features"]
    best_params  = saved.get("best_params", {})
    saved_thresh = saved.get("threshold", 0.5)

    sep(f"TEST: {panel}  |  kayıtlı esik={saved_thresh:.4f}")
    print(f"  Model : {model_path}")
    print(f"  Ozellik: {len(features)}")
    if best_params:
        print(f"  CatBoost params (Optuna):")
        for k, v in best_params.items():
            print(f"    {k:<22}: {v}")

    feat_df = load_panel_features(panel)
    X = feat_df.drop(columns=[TARGET], errors="ignore")[features]
    y = feat_df[TARGET]

    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    print(f"\n  Satir={len(y)}  POS={n_pos}  NEG={n_neg}  Oran={y.mean():.3f}")

    n_splits = min(5, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_prob = np.zeros(len(y))

    sep(f"OOF CROSS-VALIDATION  ({n_splits} FOLD | Ensemble, Optuna yok)")
    print(f"  {'Fold':>4}  {'AUC':>6}  {'MCC':>6}  {'MacroF1':>7}  {'F1-NEG':>6}  {'F1-POS':>6}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        p = dict(best_params)
        p.update({"auto_class_weights": "Balanced", "eval_metric": "F1",
                   "random_seed": RANDOM_STATE, "verbose": 0})
        cb  = CatBoostClassifier(**p)
        lgb = LGBMClassifier(n_estimators=300, max_depth=4, reg_lambda=5,
                             learning_rate=0.05, class_weight="balanced",
                             random_state=RANDOM_STATE, verbose=-1)
        xgb = XGBClassifier(n_estimators=300, max_depth=4, reg_lambda=10,
                            learning_rate=0.05,
                            scale_pos_weight=n_neg / max(n_pos, 1),
                            eval_metric="logloss", random_state=RANDOM_STATE,
                            verbosity=0, n_jobs=-1)

        cb.fit(X_tr, y_tr); lgb.fit(X_tr, y_tr); xgb.fit(X_tr, y_tr)
        prob = ensemble_prob([cb, lgb, xgb], X_te)
        all_prob[te_idx] = prob

        pred = (prob >= saved_thresh).astype(int)
        try:
            auc = roc_auc_score(y_te, prob)
        except ValueError:
            auc = float("nan")
        mcc = matthews_corrcoef(y_te, pred)
        mf1 = f1_score(y_te, pred, average="macro", zero_division=0)
        f1n = f1_score(y_te, pred, pos_label=0,     zero_division=0)
        f1p = f1_score(y_te, pred, pos_label=1,     zero_division=0)
        print(f"  {fold:>4}  {auc:>6.4f}  {mcc:>6.4f}  {mf1:>7.4f}  {f1n:>6.4f}  {f1p:>6.4f}")

    # Dinamik eşik yeniden hesapla
    sep("ESIK KONTROLU  (OOF üzerinde döngü)")
    dyn_thresh, dyn_mf1 = find_best_threshold(y.values, all_prob)
    print(f"\n  Kayitli esik (egitimden) : {saved_thresh:.4f}")
    print(f"  Yeni OOF esigi (test)    : {dyn_thresh:.4f}  (MacroF1={dyn_mf1:.4f})")

    threshold = saved_thresh  # Test tutarlılığı için kayıtlı eşiği kullan

    all_pred = (all_prob >= threshold).astype(int)
    oof_auc = roc_auc_score(y.values, all_prob)
    oof_mcc = matthews_corrcoef(y.values, all_pred)
    oof_mf1 = f1_score(y.values, all_pred, average="macro",   zero_division=0)
    oof_f1n = f1_score(y.values, all_pred, pos_label=0,       zero_division=0)
    oof_f1p = f1_score(y.values, all_pred, pos_label=1,       zero_division=0)

    sep("GENEL OOF METRIKLERI")
    print(f"\n  {'Metrik':<24}  {'Deger':>8}")
    print(f"  {'-'*24}  {'-'*8}")
    print(f"  {'ROC-AUC':<24}  {oof_auc:>8.4f}")
    print(f"  {'MCC (Matthews Corr.)':<24}  {oof_mcc:>8.4f}  <- dengeli sinif metrigi")
    print(f"  {'Macro F1':<24}  {oof_mf1:>8.4f}  <- yarisma hedefi")
    print(f"  {'F1 Benign (NEG)':<24}  {oof_f1n:>8.4f}")
    print(f"  {'F1 Patojenik (POS)':<24}  {oof_f1p:>8.4f}")

    fpr, tpr, thresholds = roc_curve(y.values, all_prob)
    sep(f"ROC EGRISI  (AUC = {oof_auc:.4f})")
    print_roc_table(fpr, tpr, thresholds)

    cm = confusion_matrix(y.values, all_pred)
    tn, fp, fn, tp = cm.ravel()
    sep("CONFUSION MATRIX")
    print(f"\n  {'':20s}  Pred-NEG  Pred-POS")
    print(f"  {'Gercek Benign (0)':<20}  {tn:>8d}  {fp:>8d}  (n={n_neg})")
    print(f"  {'Gercek Patojenik (1)':<20}  {fn:>8d}  {tp:>8d}  (n={n_pos})")

    sep("CLASSIFICATION REPORT")
    print(classification_report(y.values, all_pred,
                                target_names=["Benign (0)", "Patojenik (1)"],
                                zero_division=0))

    cb_final = final_models[0]
    imp = pd.Series(cb_final.feature_importances_, index=features).sort_values(ascending=False)
    sep("FEATURE IMPORTANCE - TOP 20  (final CatBoost, tum veri)")
    print_importance_table(imp, top_n=20)

    print(f"\n{'=' * 62}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim : python src/test_panel.py <PANEL>")
        print("Ornek    : python src/test_panel.py CFTR")
        sys.exit(1)
    test_panel(sys.argv[1].upper())
