"""
test_master.py - MASTER paneli için test seti değerlendirmesi
  • Kaydedilmiş model_MASTER.pkl yükler
  • YARISMA_TEST.csv üzerine aynı özellik mühendisliği pipeline'ını uygular
  • NaN doldurma için eğitim medyanlarını kullanır (data leakage yok)
  • AUC, MCC, Macro F1, Confusion Matrix raporlar
  • Tahminleri outputs/predictions_MASTER.csv'ye yazar

Kullanım:
    python src/test_master.py
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    classification_report, matthews_corrcoef,
)
warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PANEL       = "MASTER"
TARGET      = "Label"


def sep(title=""):
    print(f"\n{'=' * 62}")
    if title:
        print(f"  {title}")
        print("=" * 62)


def preprocess(raw_df: pd.DataFrame, train_raw: pd.DataFrame,
               train_medians: pd.Series, feature_cols: list) -> pd.DataFrame:
    """
    Eğitim pipeline'ı ile aynı dönüşümleri test verisine uygular.
    NaN doldurma için eğitim medyanları kullanılır.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    df = raw_df.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")

    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes

    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

    df = add_bio_features(df, raw_df)

    # Tamamen boş ve %80+ NaN kolonları düşür
    df = df.dropna(axis=1, how='all')
    label_col = df.pop(TARGET) if TARGET in df.columns else None
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2))

    # NaN doldurma: eğitim medyanları (data leakage yok)
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if df[col].isna().any():
            fill_val = train_medians.get(col, 0)
            df[col] = df[col].fillna(fill_val)

    if label_col is not None:
        df[TARGET] = label_col.values

    # Eğitim feature listesiyle hizala: eksik → 0, fazla → at
    feat_no_label = [c for c in feature_cols if c != TARGET]
    for col in feat_no_label:
        if col not in df.columns:
            df[col] = 0
    df = df[feat_no_label + ([TARGET] if TARGET in df.columns else [])]

    return df


def ensemble_prob(models: list, X: pd.DataFrame) -> np.ndarray:
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return probs.mean(axis=1)


def main():
    sep(f"PANEL: {PANEL} - Test Seti Degerlendirmesi")

    # ── Model yükle ───────────────────────────────────────────
    model_path = OUTPUTS_DIR / f"model_{PANEL}.pkl"
    if not model_path.exists():
        print(f"  [HATA] Model bulunamadi: {model_path}")
        print(f"  Once 'python src/train_master.py' calistirin.")
        return

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    models     = bundle["models"]
    threshold  = bundle["threshold"]
    features   = bundle["features"]
    print(f"  Model     : {model_path.name}")
    print(f"  Esik      : {threshold:.4f}")
    print(f"  Ozellikler: {len(features)}")
    print(f"  Alt modeller: {[type(m).__name__ for m in models]}")

    # ── Eğitim medyanlarını hesapla ───────────────────────────
    sep("Egitim Medyanlari Hesaplaniyor")
    train_raw = pd.read_csv(RAW_DIR / f"YARISMA_TRAIN_{PANEL}_2.csv", low_memory=False)
    print(f"  Egitim seti: {train_raw.shape}")

    # Eğitim preprocessing (medyan için)
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    tr = train_raw.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")
    for col in [c for c in tr.columns if c.startswith("CAT_")]:
        tr[col] = pd.Categorical(tr[col]).codes
    aa_cols = [c for c in tr.columns if c.startswith("AA_")]
    if aa_cols:
        tr = pd.get_dummies(tr, columns=aa_cols, drop_first=True)
    tr[tr.select_dtypes("bool").columns] = tr.select_dtypes("bool").astype(int)
    tr = add_bio_features(tr, train_raw)
    tr = tr.dropna(axis=1, how='all')
    if TARGET in tr.columns:
        tr = tr.drop(columns=[TARGET])
    tr = tr.dropna(axis=1, thresh=int(len(tr) * 0.2))
    train_medians = tr.select_dtypes(include='number').median()
    print(f"  Medyan hesaplanan kolon: {len(train_medians)}")

    # ── Test verisini yükle ve işle ───────────────────────────
    sep("Test Verisi Onislem")
    test_raw = pd.read_csv(RAW_DIR / "YARISMA_TEST.csv", low_memory=False)
    print(f"  Test seti ham: {test_raw.shape}")

    has_label = TARGET in test_raw.columns
    test_df = preprocess(test_raw, train_raw, train_medians, features)

    if has_label:
        y_true = test_df.pop(TARGET).values
        X_test = test_df
    else:
        y_true = None
        X_test = test_df

    print(f"  Test seti islenmis: {X_test.shape}")
    if y_true is not None:
        unique, counts = np.unique(y_true, return_counts=True)
        print(f"  Gercek etiketler: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    # ── Tahmin ───────────────────────────────────────────────
    sep("TAHMIN")
    y_prob = ensemble_prob(models, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    pos_count = int(y_pred.sum())
    neg_count = int((y_pred == 0).sum())
    print(f"  Tahmin Patojenik (1): {pos_count}")
    print(f"  Tahmin Benign   (0): {neg_count}")

    # ── Metrikler (Label varsa) ───────────────────────────────
    if y_true is not None:
        sep("TEST METRIKLERI  (Ensemble + Dinamik Esik)")

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        mcc   = matthews_corrcoef(y_true, y_pred)
        mf1   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        f1n   = f1_score(y_true, y_pred, pos_label=0,       zero_division=0)
        f1p   = f1_score(y_true, y_pred, pos_label=1,       zero_division=0)

        # 0.50 sabit eşik karşılaştırma
        pred_05 = (y_prob >= 0.50).astype(int)
        mf1_05  = f1_score(y_true, pred_05, average="macro", zero_division=0)

        print(f"\n  {'Metrik':<24}  {'Deger':>8}")
        print(f"  {'-'*24}  {'-'*8}")
        print(f"  {'ROC-AUC':<24}  {auc:>8.4f}")
        print(f"  {'MCC':<24}  {mcc:>8.4f}")
        print(f"  {'Macro F1':<24}  {mf1:>8.4f}  <-- hedef")
        print(f"  {'F1 Benign (NEG)':<24}  {f1n:>8.4f}")
        print(f"  {'F1 Patojenik (POS)':<24}  {f1p:>8.4f}")
        print(f"  {'Esik':<24}  {threshold:>8.4f}")
        print(f"\n  Karsilastirma:")
        print(f"    Sabit 0.50        -> Macro F1 = {mf1_05:.4f}")
        print(f"    Dinamik {threshold:.4f}  -> Macro F1 = {mf1:.4f}  ({mf1-mf1_05:+.4f})")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        n_neg = int((y_true == 0).sum())
        n_pos = int((y_true == 1).sum())
        print(f"\n  Confusion Matrix:  [esik={threshold:.4f}]")
        print(f"  {'':20s}  Pred-NEG  Pred-POS")
        print(f"  {'Gercek Benign (0)':<20}  {tn:>8d}  {fp:>8d}  (n={n_neg})")
        print(f"  {'Gercek Patojenik (1)':<20}  {fn:>8d}  {tp:>8d}  (n={n_pos})")

        print(f"\n  Classification Report:  [esik={threshold:.4f}]")
        print(classification_report(y_true, y_pred,
                                    target_names=["Benign (0)", "Patojenik (1)"],
                                    zero_division=0))

        # Sonuçları JSON'a kaydet
        result = {
            "panel": PANEL, "split": "TEST",
            "threshold": round(threshold, 4),
            "auc":      round(float(auc), 4),
            "mcc":      round(float(mcc), 4),
            "macro_f1": round(float(mf1), 4),
            "f1_neg":   round(float(f1n), 4),
            "f1_pos":   round(float(f1p), 4),
        }
        METRICS_DIR.mkdir(exist_ok=True)
        out_json = METRICS_DIR / f"panel_{PANEL}_test_metrics.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Metrikler kaydedildi: {out_json}")

    # ── Tahminleri kaydet ─────────────────────────────────────
    sep("TAHMINLER KAYDEDILIYOR")
    variant_ids = test_raw["Variant_ID"].values if "Variant_ID" in test_raw.columns else range(len(y_pred))
    out_df = pd.DataFrame({
        "Variant_ID": variant_ids,
        "prob_patojenik": y_prob.round(4),
        "pred_label": y_pred,
    })
    if y_true is not None:
        out_df["true_label"] = y_true
        out_df["correct"] = (y_pred == y_true).astype(int)

    pred_path = OUTPUTS_DIR / f"predictions_{PANEL}.csv"
    out_df.to_csv(pred_path, index=False)
    print(f"  {len(out_df)} tahmin kaydedildi: {pred_path}")

    sep("TAMAMLANDI")
    if y_true is not None:
        print(f"  TEST  AUC={auc:.4f}  MCC={mcc:.4f}  MacroF1={mf1:.4f}  Esik={threshold:.4f}")
    print()


if __name__ == "__main__":
    main()
