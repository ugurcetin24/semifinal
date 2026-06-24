"""
train_master.py - MASTER paneli için tam eğitim scripti
  • Biyolojik özellikler: BLOSUM62, PAM250 amino asit ikame skorları,
    Grantham mesafesi, Kyte-Doolittle hidrofobiklik
  • BIO_missing_count - satır bazlı NaN sayısı (nadir varyant sinyali)
  • EK_ evrimsel korunmuşluk: mean / max / std
  • AL_ alel frekansı: global mean/max/std + 5'erli popülasyon blokları
  • Optuna Bayesian optimizasyon - CatBoost hiperparametre arama
  • Ensemble soft voting: CatBoost + LightGBM + XGBoost
  • Eşik: OOF olasılıkları üzerinde Macro F1 maksimize eden döngü (0.20-0.80)

Kullanım:
    python src/train_master.py
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    classification_report, matthews_corrcoef,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

PANEL         = "MASTER"   # <-- sadece bu satır her dosyada değişir
ROOT      = Path(__file__).parent.parent
RAW_DIR   = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR   = OUTPUTS_DIR / "metrics"
TARGET        = "Label"
N_FOLDS       = 5
RANDOM_STATE  = 42
N_TRIALS      = 40
TOP_N         = 20


def sep(title=""):
    print(f"\n{'=' * 62}")
    if title:
        print(f"  {title}")
        print("=" * 62)


def load_features():
    """
    Ham veriyi okur, CAT_/AA_ kodlar, bio_features ile
    BLOSUM62, PAM250, Grantham, EK_/AL_ istatistiklerini ekler.
    NaN'lar korunur - CatBoost içsel olarak işler.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    raw_p = pd.read_csv(RAW_DIR / f"YARISMA_TRAIN_{PANEL}.csv", low_memory=False)
    df    = raw_p.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")

    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes

    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

    df = add_bio_features(df, raw_p)
    return df


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        lo: float = 0.20, hi: float = 0.80,
                        steps: int = 121) -> tuple:
    """OOF olasılıkları üzerinde Macro F1'i maksimize eden eşiği döngüyle arar."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(lo, hi, steps):
        score = f1_score(y_true, (y_prob >= t).astype(int),
                         average="macro", zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def build_catboost(params: dict) -> CatBoostClassifier:
    p = dict(params)
    p.update({"auto_class_weights": "Balanced", "eval_metric": "F1",
               "random_seed": RANDOM_STATE, "verbose": 0})
    return CatBoostClassifier(**p)


def build_lgbm(n_samples: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300 if n_samples < 200 else 500,
        max_depth=4, reg_lambda=5, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1,
    )


def build_xgb(n_neg: int, n_pos: int, n_samples: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300 if n_samples < 200 else 500,
        max_depth=4, reg_lambda=10, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss", random_state=RANDOM_STATE,
        verbosity=0, n_jobs=-1,
    )


def ensemble_prob(models: list, X: pd.DataFrame) -> np.ndarray:
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return probs.mean(axis=1)


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
    sep(f"PANEL: {PANEL} - Tam Egitim Pipeline")

    feat_df = load_features()
    X = feat_df.drop(columns=[TARGET], errors="ignore")
    y = feat_df[TARGET]

    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    print(f"  Satirlar   : {len(y)}")
    print(f"  POS (1)    : {n_pos}   NEG (0): {n_neg}   Oran: {y.mean():.3f}")
    print(f"  Ozellikler : {X.shape[1]}")

    if n_neg < 5:
        print(f"  [UYARI] Cok az negatif ornek ({n_neg}), egitim iptal.")
        return

    # ── Optuna: CatBoost hiperparametre arama ─────────────────
    sep(f"OPTUNA - CatBoost Bayesian Optimizasyon ({N_TRIALS} deneme)")
    print("  Arama: ", end="", flush=True)

    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 100, 600),
            "depth":               trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1, 15),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "auto_class_weights": "Balanced",
            "eval_metric": "F1", "random_seed": RANDOM_STATE, "verbose": 0,
        }
        fold_probs = np.zeros(len(y))
        for tr, val in skf3.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            fold_probs[val] = m.predict_proba(X.iloc[val])[:, 1]
        # Optuna içinde de dinamik eşik arama döngüsü
        _, score = find_best_threshold(y.values, fold_probs)
        return score

    def _progress(study, trial):
        if trial.number % 10 == 9:
            print(f"[{trial.number+1}ok best={study.best_value:.4f}]", end=" ", flush=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_progress])
    print()

    best_params = study.best_params
    print(f"\n  En iyi Macro F1 (3-fold, dinamik esik): {study.best_value:.4f}")
    print(f"  En iyi parametreler:")
    for k, v in best_params.items():
        print(f"    {k:<22}: {v}")

    # ── 5-Fold OOF Ensemble ───────────────────────────────────
    n_splits = min(N_FOLDS, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_prob       = np.zeros(len(y))
    fold_imps      = []
    fold_rows      = []

    sep(f"{n_splits}-FOLD OOF - ENSEMBLE  [CatBoost + LightGBM + XGBoost]")
    print(f"  {'Fold':>4}  {'AUC':>6}  {'F1-NEG':>6}  {'F1-POS':>6}  Top-5 ozellik")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*40}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        cb  = build_catboost(best_params)
        lgb = build_lgbm(len(X_tr))
        xgb = build_xgb(n_neg, n_pos, len(X_tr))

        cb.fit(X_tr, y_tr); lgb.fit(X_tr, y_tr); xgb.fit(X_tr, y_tr)

        prob = ensemble_prob([cb, lgb, xgb], X_te)
        all_prob[te_idx] = prob

        try:
            auc = roc_auc_score(y_te, prob)
        except ValueError:
            auc = float("nan")

        pred_05 = (prob >= 0.5).astype(int)
        f1n_05  = f1_score(y_te, pred_05, pos_label=0, zero_division=0)
        f1p_05  = f1_score(y_te, pred_05, pos_label=1, zero_division=0)

        imp = pd.Series(cb.feature_importances_, index=X.columns).sort_values(ascending=False)
        fold_imps.append(imp)
        top5 = ", ".join(f"{c}({v:.1f}%)"
                         for c, v in zip(imp.head(5).index,
                                         imp.head(5).values / imp.sum() * 100))

        print(f"  {fold:>4}  {auc:>6.4f}  {f1n_05:>6.4f}  {f1p_05:>6.4f}  {top5}")
        fold_rows.append({"fold": fold, "auc": auc, "f1_neg_05": f1n_05, "f1_pos_05": f1p_05})

    # ── Dinamik eşik arama döngüsü ────────────────────────────
    sep("ESIK BELIRLEME  (OOF Macro F1 Maksimizasyonu - 121 adim)")
    threshold, thresh_mf1 = find_best_threshold(y.values, all_prob)

    t05_mf1 = f1_score(y.values, (all_prob >= 0.50).astype(int),
                       average="macro", zero_division=0)

    print(f"\n  Arama araligi : 0.20 – 0.80  (121 adim)")
    print(f"  Bulunan esik  : {threshold:.4f}")
    print(f"  Macro F1 (OOF): {thresh_mf1:.4f}")
    print(f"\n  Karsilastirma:")
    print(f"    Sabit 0.50   -> Macro F1 = {t05_mf1:.4f}")
    print(f"    Dinamik {threshold:.4f} -> Macro F1 = {thresh_mf1:.4f}  (+{thresh_mf1 - t05_mf1:.4f})")

    # ── OOF Metrikleri ────────────────────────────────────────
    all_pred = (all_prob >= threshold).astype(int)
    oof_auc  = roc_auc_score(y.values, all_prob)
    oof_mcc  = matthews_corrcoef(y.values, all_pred)
    oof_mf1  = f1_score(y.values, all_pred, average="macro",   zero_division=0)
    oof_f1n  = f1_score(y.values, all_pred, pos_label=0,       zero_division=0)
    oof_f1p  = f1_score(y.values, all_pred, pos_label=1,       zero_division=0)

    sep("OOF TEST METRIKLERI  (Ensemble + Dinamik Esik)")
    print(f"\n  {'Metrik':<24}  {'Deger':>8}")
    print(f"  {'-'*24}  {'-'*8}")
    print(f"  {'ROC-AUC':<24}  {oof_auc:>8.4f}")
    print(f"  {'MCC':<24}  {oof_mcc:>8.4f}")
    print(f"  {'Macro F1':<24}  {oof_mf1:>8.4f}  <-- hedef")
    print(f"  {'F1 Benign (NEG)':<24}  {oof_f1n:>8.4f}")
    print(f"  {'F1 Patojenik (POS)':<24}  {oof_f1p:>8.4f}")
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
    sep(f"OZELLIK ONEMLERI - TOP {TOP_N}  (CatBoost fold ortalaması)")
    print_importance_table(avg_imp, top_n=TOP_N)

    # ── Final model - tüm veri ────────────────────────────────
    sep("FINAL MODEL  (tum veri ile egitim)")
    final_cb  = build_catboost(best_params)
    final_lgb = build_lgbm(len(X))
    final_xgb = build_xgb(n_neg, n_pos, len(X))
    final_cb.fit(X, y); final_lgb.fit(X, y); final_xgb.fit(X, y)
    print("  3 model egitildi (CatBoost, LightGBM, XGBoost).")

    # ── Kaydet ───────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(exist_ok=True); METRICS_DIR.mkdir(exist_ok=True)

    model_path = OUTPUTS_DIR / f"model_{PANEL}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models":      [final_cb, final_lgb, final_xgb],
            "threshold":   threshold,
            "features":    list(X.columns),
            "best_params": best_params,
            "panel":       PANEL,
        }, f)

    result = {
        "panel": PANEL, "threshold": round(threshold, 4),
        "oof_auc":  round(float(oof_auc), 4),
        "oof_mcc":  round(float(oof_mcc), 4),
        "macro_f1": round(oof_mf1, 4),
        "f1_neg":   round(oof_f1n, 4),
        "f1_pos":   round(oof_f1p, 4),
        "best_params": best_params,
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
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    train()
