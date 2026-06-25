"""
analyze_panel.py  –  TEKNOFEST 2026 Sağlıkta Yapay Zeka
Kapsamlı panel analizi: performans metrikleri, SHAP açıklanabilirliği,
hiper-parametre duyarlılık analizi ve model karşılaştırması.

Çıktılar  outputs/figures/{PANEL}/:
  {PANEL}_1_performance.png   — Confusion matrix, ROC, PR, eşik analizi
  {PANEL}_2_shap.png          — SHAP beeswarm + bar (tüm eğitim seti)
  {PANEL}_3_shap_waterfall.png— SHAP waterfall (FP + TP + TN örneği)
  {PANEL}_4_comparison.png    — Model × fold karşılaştırması
  {PANEL}_5_hyperparams.png   — Parametre tablosu + duyarlılık ısı haritası

Kullanım:
    python src/analyze_panel.py PAH
    python src/analyze_panel.py CFTR
    python src/analyze_panel.py KANSER
    python src/analyze_panel.py MASTER

Gereksinimler:
    pip install shap seaborn
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, average_precision_score,
    matthews_corrcoef, recall_score, balanced_accuracy_score,
    precision_recall_curve, roc_curve, precision_score,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

warnings.filterwarnings("ignore")

# ── Yollar ────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"
TARGET      = "Label"

# ── Eğitim sabitleri (train dosyalarıyla senkron) ─────────────────
RANDOM_STATE               = 42
N_FOLDS                    = 5
BALANCED_BAGS              = 3
PATHOGENIC_TO_BENIGN_RATIO = 2.0
MIN_PATHOGENIC_RECALL      = 0.90

CAT_PARAMS = {
    "iterations": 600, "learning_rate": 0.03, "depth": 4,
    "l2_leaf_reg": 5, "loss_function": "Logloss", "eval_metric": "AUC",
    "auto_class_weights": "Balanced", "random_seed": 60,
    "verbose": 0, "allow_writing_files": False, "thread_count": -1,
}
LGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.04, "num_leaves": 15,
    "max_depth": 4, "min_child_samples": 12, "subsample": 0.80,
    "subsample_freq": 1, "colsample_bytree": 0.80,
    "reg_alpha": 0.10, "reg_lambda": 1.00, "objective": "binary",
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
}
XGB_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.04, "max_depth": 4,
    "min_child_weight": 5.0, "subsample": 0.80, "colsample_bytree": 0.80,
    "gamma": 1.0, "reg_alpha": 0.10, "reg_lambda": 2.00,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "random_state": RANDOM_STATE,
    "n_jobs": -1, "verbosity": 0,
}
ENSEMBLE_WEIGHTS = [2, 1, 1]

# ── Renk paleti ───────────────────────────────────────────────────
C = {
    "path":     "#e74c3c",   # Patojenik
    "benign":   "#3498db",   # Benign
    "ens":      "#2c3e50",   # Ensemble
    "cat":      "#8e44ad",   # CatBoost
    "lgb":      "#27ae60",   # LightGBM
    "xgb":      "#e67e22",   # XGBoost
    "thresh":   "#f39c12",   # Eşik çizgisi
    "mcc":      "#1abc9c",   # MCC
    "bg":       "#ffffff",
    "panel":    "#f8f9fa",
    "grid":     "#dee2e6",
    "text":     "#212529",
    "accent":   "#6c5ce7",
}

PANEL_CSV = {
    "PAH":    "YARISMA_TRAIN_PAH_BIRLESIK_ORJINAL_PLUS_SYNTHETIC.csv",
    "CFTR":   "YARISMA_TRAIN_CFTR_AUGMENTED.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "MASTER": "YARISMA_TRAIN_MASTER_2.csv",
}

# ─────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────
def load_features(panel: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from bio_features import add_bio_features

    csv_name = PANEL_CSV[panel]
    raw_p = pd.read_csv(RAW_DIR / csv_name, low_memory=False)
    df = raw_p.copy().drop(columns=["Variant_ID", "panel"], errors="ignore")

    for col in [c for c in df.columns if c.startswith("CAT_")]:
        df[col] = pd.Categorical(df[col]).codes
    aa_cols = [c for c in df.columns if c.startswith("AA_")]
    if aa_cols:
        df = pd.get_dummies(df, columns=aa_cols, drop_first=True)
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)
    df = add_bio_features(df, raw_p)

    if panel == "MASTER":
        df = df.dropna(axis=1, how="all")
        lbl = df.pop(TARGET) if TARGET in df.columns else None
        df = df.dropna(axis=1, thresh=int(len(df) * 0.2))
        df[df.select_dtypes("number").columns] = (
            df.select_dtypes("number").fillna(df.select_dtypes("number").median())
        )
        if lbl is not None:
            df[TARGET] = lbl.values

    return df, raw_p


def make_exact_row_groups(raw_x):
    return pd.factorize(pd.util.hash_pandas_object(raw_x, index=False), sort=False)[0]


def inverse_duplicate_weights(groups):
    s = pd.Series(groups)
    return (1.0 / s.map(s.value_counts())).to_numpy()


def make_balanced_bag_indices(y_train, bag_num, fold_num):
    bi = np.flatnonzero(y_train == 0)
    pi = np.flatnonzero(y_train == 1)
    sz = min(int(np.ceil(len(bi) * PATHOGENIC_TO_BENIGN_RATIO)), len(pi))
    rng = np.random.default_rng(RANDOM_STATE + 1000 * fold_num + 37 * bag_num)
    sel = np.concatenate([bi, rng.choice(pi, size=sz, replace=False)])
    rng.shuffle(sel)
    return sel


def build_lgbm(n_neg, n_pos):
    p = dict(LGBM_PARAMS); p["scale_pos_weight"] = n_neg / max(n_pos, 1)
    return LGBMClassifier(**p)


def build_xgb(n_neg, n_pos):
    p = dict(XGB_PARAMS); p["scale_pos_weight"] = n_neg / max(n_pos, 1)
    return XGBClassifier(**p)


def find_best_threshold(y_true, y_prob):
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
            if best is None or score > best[1]: best = entry
        if fallback is None or score > fallback[1]: fallback = entry
    return best if best is not None else fallback


# ─────────────────────────────────────────────────────────────────
# OOF değerlendirme (bireysel model takibiyle)
# ─────────────────────────────────────────────────────────────────
def run_oof(X, y, groups, dup_weights, verbose=True):
    n_neg = int((y == 0).sum())
    n_pos = int(y.sum())
    n_splits = min(N_FOLDS, n_neg)
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_ens = np.zeros(len(y)); oof_cat = np.zeros(len(y))
    oof_lgb = np.zeros(len(y)); oof_xgb = np.zeros(len(y))
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        fdw = dup_weights[tr_idx]

        cat_p, lgb_p, xgb_p = [], [], []

        for bag in range(1, BALANCED_BAGS + 1):
            bi = make_balanced_bag_indices(y_tr.values, bag, fold)
            yb = y_tr.iloc[bi]; Xb = X_tr.iloc[bi]; wb = fdw[bi]
            nb_neg = int((yb == 0).sum()); nb_pos = int((yb == 1).sum())

            cb = CatBoostClassifier(**CAT_PARAMS)
            lb = build_lgbm(nb_neg, nb_pos)
            xb = build_xgb(nb_neg, nb_pos)
            cb.fit(Xb, yb, sample_weight=wb)
            lb.fit(Xb, yb, sample_weight=wb)
            xb.fit(Xb, yb, sample_weight=wb)
            cat_p.append(cb.predict_proba(X_te)[:, 1])
            lgb_p.append(lb.predict_proba(X_te)[:, 1])
            xgb_p.append(xb.predict_proba(X_te)[:, 1])

        cm = np.mean(cat_p, axis=0); lm = np.mean(lgb_p, axis=0); xm = np.mean(xgb_p, axis=0)
        em = np.average(np.column_stack([cm, lm, xm]), axis=1, weights=ENSEMBLE_WEIGHTS)

        oof_ens[te_idx] = em; oof_cat[te_idx] = cm
        oof_lgb[te_idx] = lm; oof_xgb[te_idx] = xm

        try:
            auc_e = roc_auc_score(y_te, em); auc_c = roc_auc_score(y_te, cm)
            auc_l = roc_auc_score(y_te, lm); auc_x = roc_auc_score(y_te, xm)
        except ValueError:
            auc_e = auc_c = auc_l = auc_x = float("nan")

        p05 = (em >= 0.5).astype(int)
        f1n = f1_score(y_te, p05, pos_label=0, zero_division=0)
        f1p = f1_score(y_te, p05, pos_label=1, zero_division=0)

        fold_rows.append({
            "fold": fold,
            "auc_ens": auc_e, "auc_cat": auc_c,
            "auc_lgb": auc_l, "auc_xgb": auc_x,
            "f1_neg": f1n, "f1_pos": f1p,
        })
        if verbose:
            print(f"    Fold {fold}:  AUC={auc_e:.4f}  F1-NEG={f1n:.4f}  F1-POS={f1p:.4f}")

    return oof_ens, oof_cat, oof_lgb, oof_xgb, fold_rows


# ─────────────────────────────────────────────────────────────────
# Hiper-parametre duyarlılık analizi (depth × learning_rate)
# ─────────────────────────────────────────────────────────────────
def run_sensitivity(X, y, depths=(3, 4, 5, 6), lrs=(0.01, 0.03, 0.05, 0.10)):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    total = len(depths) * len(lrs)
    done = 0
    print(f"  Duyarlılık analizi: {total} kombinasyon × 3-fold ...")

    for depth in depths:
        for lr in lrs:
            mccs = []
            for tr, te in skf.split(X, y):
                p = dict(CAT_PARAMS)
                p.update({"depth": depth, "learning_rate": lr, "iterations": 200})
                cb = CatBoostClassifier(**p)
                cb.fit(X.iloc[tr], y.iloc[tr])
                pred = cb.predict(X.iloc[te])
                mccs.append(matthews_corrcoef(y.iloc[te], pred))
            results[(depth, lr)] = float(np.mean(mccs))
            done += 1
            print(f"    [{done}/{total}] depth={depth} lr={lr:.2f}  MCC={results[(depth,lr)]:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────
# Figure 1 — Performans Panosu
# ─────────────────────────────────────────────────────────────────
def fig1_performance(y, oof_ens, oof_cat, oof_lgb, oof_xgb, threshold, panel, save_dir):
    fig = plt.figure(figsize=(16, 12), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"])
        for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])

    y_pred = (oof_ens >= threshold).astype(int)
    y_np   = y.values

    # ── [0] Confusion Matrix ──────────────────────────────────
    ax = axes[0]
    cm = confusion_matrix(y_np, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.75, label="Oran")
    labels = ["Benign (0)", "Patojenik (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9)
    for i in range(2):
        for j in range(2):
            col = "white" if cmn[i, j] > 0.55 else C["text"]
            ax.text(j, i, f"{cmn[i,j]:.1%}\n(n={cm[i,j]})",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=col)
    ax.set_title("Karmaşıklık Matrisi", fontweight="bold", pad=8)
    ax.set_ylabel("Gerçek Sınıf"); ax.set_xlabel("Tahmin Edilen")
    ax.grid(False)

    # ── [1] ROC Eğrisi ────────────────────────────────────────
    ax = axes[1]
    model_probs = [
        (oof_ens, "Ensemble", C["ens"],  2.5, "-"),
        (oof_cat, "CatBoost", C["cat"],  1.5, "--"),
        (oof_lgb, "LightGBM", C["lgb"],  1.5, "--"),
        (oof_xgb, "XGBoost",  C["xgb"],  1.5, "--"),
    ]
    for probs, lbl, col, lw, ls in model_probs:
        fpr, tpr, _ = roc_curve(y_np, probs)
        auc = roc_auc_score(y_np, probs)
        ax.plot(fpr, tpr, color=col, lw=lw, ls=ls, label=f"{lbl}  AUC={auc:.3f}")
    ax.fill_between(*roc_curve(y_np, oof_ens)[:2], alpha=0.07, color=C["ens"])
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4, label="Rastgele")
    ax.set_xlabel("Yanlış Pozitif Oranı (1-Özgüllük)")
    ax.set_ylabel("Doğru Pozitif Oranı (Duyarlılık)")
    ax.set_title("ROC Eğrisi", fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    # ── [2] Kesinlik–Duyarlılık Eğrisi ───────────────────────
    ax = axes[2]
    prev = y_np.mean()
    for probs, lbl, col, lw, ls in model_probs:
        prec, rec, _ = precision_recall_curve(y_np, probs)
        ap = average_precision_score(y_np, probs)
        ax.plot(rec, prec, color=col, lw=lw, ls=ls, label=f"{lbl}  AP={ap:.3f}")
    ax.fill_between(*precision_recall_curve(y_np, oof_ens)[:2][::-1], alpha=0.07, color=C["ens"])
    ax.axhline(prev, color="gray", lw=0.8, ls=":", label=f"Baseline={prev:.2f}")
    ax.set_xlabel("Duyarlılık (Recall)"); ax.set_ylabel("Kesinlik (Precision)")
    ax.set_title("Kesinlik–Duyarlılık Eğrisi", fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.08)

    # ── [3] Eşik Duyarlılık Analizi ──────────────────────────
    ax = axes[3]
    ts = np.linspace(0.05, 0.95, 181)
    f1s, mccs, prs, brs = [], [], [], []
    for t in ts:
        p = (oof_ens >= t).astype(int)
        f1s.append(f1_score(y_np, p, average="macro",  zero_division=0))
        mccs.append((matthews_corrcoef(y_np, p) + 1) / 2)
        prs.append(recall_score(y_np, p, pos_label=1,  zero_division=0))
        brs.append(recall_score(y_np, p, pos_label=0,  zero_division=0))
    ax.plot(ts, f1s,  color=C["ens"],    lw=2.5,        label="Macro F1")
    ax.plot(ts, mccs, color=C["mcc"],    lw=2.5, ls="--", label="MCC (norm.)")
    ax.plot(ts, prs,  color=C["path"],   lw=1.8, ls=":",  label="Patojenik Recall")
    ax.plot(ts, brs,  color=C["benign"], lw=1.8, ls=":",  label="Benign Recall")
    ax.axvline(threshold, color=C["thresh"], lw=2.2, ls="-.",
               label=f"Seçilen eşik = {threshold:.2f}")
    ax.axhline(MIN_PATHOGENIC_RECALL, color=C["path"], lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("Eşik Değeri"); ax.set_ylabel("Metrik")
    ax.set_title("Eşik Duyarlılık Analizi", fontweight="bold", pad=8)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_xlim(0.05, 0.95); ax.set_ylim(-0.02, 1.10)

    # ── Özet şeridi ───────────────────────────────────────────
    f1v   = f1_score(y_np, y_pred, average="macro", zero_division=0)
    mccv  = matthews_corrcoef(y_np, y_pred)
    apv   = average_precision_score(y_np, oof_ens)
    aucv  = roc_auc_score(y_np, oof_ens)
    pathR = recall_score(y_np, y_pred, pos_label=1, zero_division=0)
    benR  = recall_score(y_np, y_pred, pos_label=0, zero_division=0)

    summary = (f"Macro F1 = {f1v:.4f}   |   MCC = {mccv:.4f}   |   "
               f"AP (PR-AUC) = {apv:.4f}   |   ROC-AUC = {aucv:.4f}   |   "
               f"PathRecall = {pathR:.4f}   |   BenignRecall = {benR:.4f}")
    fig.text(0.5, 0.005, summary, ha="center", fontsize=9.5, fontweight="bold",
             color=C["text"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf4fb",
                       edgecolor=C["benign"], lw=1.5))

    fig.suptitle(
        f"PANEL: {panel}  —  Performans Panosu  "
        f"[OOF {N_FOLDS}-fold | {BALANCED_BAGS} balanced bag | eşik={threshold:.3f}]",
        fontsize=14, fontweight="bold", y=1.01, color=C["text"])

    path = save_dir / f"{panel}_1_performance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [1/5] Performans panosu kaydedildi.")
    return {"f1": f1v, "mcc": mccv, "ap": apv, "auc": aucv,
            "path_recall": pathR, "benign_recall": benR, "threshold": threshold}


# ─────────────────────────────────────────────────────────────────
# Figure 2 — SHAP Beeswarm + Bar
# ─────────────────────────────────────────────────────────────────
def fig2_shap(final_cb, X, panel, save_dir, max_display=18):
    if not SHAP_OK:
        print("  [2/5] SHAP atlandı (paket yüklü değil).")
        return

    print("  SHAP değerleri hesaplanıyor...")
    sample_size = min(len(X), 200)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=sample_size, replace=False)
    X_s = X.iloc[idx].reset_index(drop=True)

    explainer  = shap.TreeExplainer(final_cb)
    shap_vals  = explainer.shap_values(X_s)       # (n, p) numpy array
    base_value = explainer.expected_value

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    feat_names = list(X.columns)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[-max_display:][::-1]

    # Feature grup renkleri
    def feat_color(name):
        if "alblk" in name: return "#fd79a8"
        if name.startswith("BIO_"): return "#00b894"
        if name.startswith("EK_"):  return "#6c5ce7"
        if name.startswith("AL_"):  return "#fdcb6e"
        return "#74b9ff"

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=C["bg"])
    fig.subplots_adjust(wspace=0.40)

    # ── Sol: Beeswarm ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(C["panel"])
    cmap = plt.cm.RdBu_r

    for rank, fi in enumerate(reversed(top_idx)):
        y_pos  = max_display - 1 - rank
        svals  = shap_vals[:, fi]
        fvals  = X_s.iloc[:, fi].values.astype(float)
        fmin, fmax = np.nanpercentile(fvals, [5, 95])
        fnorm = np.clip((fvals - fmin) / (fmax - fmin + 1e-9), 0, 1)
        jitter = np.random.default_rng(fi).uniform(-0.20, 0.20, len(svals))
        ax.scatter(svals, y_pos + jitter, c=fnorm, cmap="RdBu_r",
                   s=14, alpha=0.70, rasterized=True, vmin=0, vmax=1)

    ax.set_yticks(range(max_display))
    ax.set_yticklabels(
        [feat_names[fi] for fi in reversed(top_idx)],
        fontsize=8.5, color=C["text"])
    ax.axvline(0, color=C["text"], lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("SHAP Değeri  (model çıktısı üzerindeki etki)", fontsize=10)
    ax.set_title("SHAP Beeswarm  (özellik değeri → renk)", fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.7)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.55, aspect=20, pad=0.02)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["Düşük", "Orta", "Yüksek"])
    cb.set_label("Özellik Değeri", fontsize=8)

    # ── Sağ: Bar Chart ────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(C["panel"])
    bar_idx = np.argsort(mean_abs)[-max_display:]
    bar_vals  = mean_abs[bar_idx]
    bar_names = [feat_names[i] for i in bar_idx]
    bar_colors = [feat_color(n) for n in bar_names]

    bars = ax.barh(range(len(bar_idx)), bar_vals, color=bar_colors,
                   edgecolor="white", height=0.7, linewidth=0.5)
    ax.set_yticks(range(len(bar_idx)))
    ax.set_yticklabels(bar_names, fontsize=8.5)
    ax.set_xlabel("Ortalama |SHAP Değeri|", fontsize=10)
    ax.set_title("SHAP Özellik Önemi  (|SHAP| ortalaması)", fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.7)

    for bar, val in zip(bars, bar_vals):
        ax.text(val + max(bar_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5, color=C["text"])

    legend_patches = [
        mpatches.Patch(color="#fd79a8", label="AL Blok (BIO)"),
        mpatches.Patch(color="#00b894", label="BIO_ (Amino asit / mesafe)"),
        mpatches.Patch(color="#6c5ce7", label="EK_ (Evrimsel korunmuşluk)"),
        mpatches.Patch(color="#fdcb6e", label="AL_ (Alel frekansı)"),
        mpatches.Patch(color="#74b9ff", label="Diğer"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor=C["grid"])

    fig.suptitle(f"PANEL: {panel}  —  SHAP Açıklanabilirlik Analizi  "
                 f"(n={sample_size} örnek, CatBoost)",
                 fontsize=14, fontweight="bold", y=1.01, color=C["text"])

    path = save_dir / f"{panel}_2_shap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [2/5] SHAP beeswarm + bar kaydedildi.")

    # ── Waterfall: FP + TP + TN ───────────────────────────────
    _fig3_shap_waterfall(shap_vals, base_value, X_s, feat_names, panel, save_dir)


def _fig3_shap_waterfall(shap_vals, base_val, X_s, feat_names, panel, save_dir):
    """SHAP waterfall için 3 temsili örnek seçer ve çizer."""

    def custom_waterfall(ax, sv, fvals, fn, title, pred_label, max_d=12):
        top = np.argsort(np.abs(sv))[-max_d:]
        top = top[np.argsort(sv[top])[::-1]]
        names = [fn[i] for i in top]; vals = sv[top]
        fv    = [fvals[fn[i]] if fn[i] in fvals.index else np.nan for i in top]

        cum = float(base_val)
        for rank, (v, n) in enumerate(zip(vals, names)):
            col  = C["path"] if v > 0 else C["benign"]
            ax.barh(rank, v, left=cum, color=col, alpha=0.85,
                    height=0.60, edgecolor="white", lw=0.5)
            ax.text(cum + v + (max(np.abs(vals)) * 0.02 * np.sign(v)),
                    rank, f"{v:+.3f}", va="center",
                    ha="left" if v > 0 else "right",
                    fontsize=7.5, fontweight="bold", color=col)
            cum += v

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [f"{n} = {v:.3g}" if isinstance(v, (int, float)) and not np.isnan(v)
             else n for n, v in zip(names, fv)],
            fontsize=8)
        ax.axvline(base_val, color="gray", lw=0.7, ls="--", alpha=0.5,
                   label=f"Baz değer = {base_val:.3f}")
        ax.axvline(cum, color=C["thresh"], lw=1.5, ls="-.",
                   label=f"Tahmin = {cum:.3f}")
        ax.set_xlabel("SHAP Katkısı"); ax.set_title(title, fontweight="bold", pad=6)
        ax.legend(fontsize=8); ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        rp = mpatches.Patch(color=C["path"],   label="Patojenikliğe iter (+)")
        bp = mpatches.Patch(color=C["benign"], label="Benignliğe iter (−)")
        ax.legend(handles=[rp, bp], loc="lower right", fontsize=7.5)

    # Gerçek etiketler bilinmiyor (unsupervised SHAP örnek seçimi)
    # En yüksek SHAP norm değerine sahip 3 örnek gösterilir
    max_shap_idx = np.argsort(np.abs(shap_vals).sum(axis=1))[-3:][::-1]

    titles = [
        "En Etkili Örnek 1  (en yüksek toplam SHAP)",
        "En Etkili Örnek 2",
        "En Etkili Örnek 3  (neden bu karar verildi?)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor=C["bg"])
    fig.subplots_adjust(wspace=0.38)

    for ax, si, title in zip(axes, max_shap_idx, titles):
        ax.set_facecolor(C["panel"])
        custom_waterfall(ax, shap_vals[si], X_s.iloc[si], feat_names, title,
                         "Patojenik" if shap_vals[si].sum() > 0 else "Benign")

    fig.suptitle(f"PANEL: {panel}  —  SHAP Waterfall Analizi  "
                 f"(Hangi özellikler bu kararı yönlendirdi?)",
                 fontsize=14, fontweight="bold", y=1.01, color=C["text"])

    path = save_dir / f"{panel}_3_shap_waterfall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [3/5] SHAP waterfall kaydedildi.")


# ─────────────────────────────────────────────────────────────────
# Figure 4 — Model × Fold Karşılaştırması
# ─────────────────────────────────────────────────────────────────
def fig4_comparison(y, oof_ens, oof_cat, oof_lgb, oof_xgb, fold_rows, panel, save_dir):
    folds = [r["fold"] for r in fold_rows]
    fig   = plt.figure(figsize=(18, 11), facecolor=C["bg"])
    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    axes  = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor(C["panel"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=C["grid"], lw=0.7)

    y_np = y.values

    # ── [0] Fold × AUC çizgi grafiği ─────────────────────────
    ax = axes[0]
    for key, lbl, col, lw, ls in [
        ("auc_ens", "Ensemble", C["ens"],  2.5, "-"),
        ("auc_cat", "CatBoost", C["cat"],  1.5, "--"),
        ("auc_lgb", "LightGBM", C["lgb"],  1.5, "--"),
        ("auc_xgb", "XGBoost",  C["xgb"],  1.5, "--"),
    ]:
        vals = [r[key] for r in fold_rows]
        ax.plot(folds, vals, "o-", color=col, lw=lw, ls=ls,
                markersize=7, label=f"{lbl} (ort={np.mean(vals):.3f})")
    ax.set_xlabel("Fold"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Fold × ROC-AUC Karşılaştırması", fontweight="bold")
    ax.legend(fontsize=8); ax.set_xticks(folds)

    # ── [1] Fold × F1-NEG çizgi grafiği ──────────────────────
    ax = axes[1]
    f1n_vals = [r["f1_neg"] for r in fold_rows]
    f1p_vals = [r["f1_pos"] for r in fold_rows]
    ax.plot(folds, f1n_vals, "s-", color=C["benign"], lw=2, markersize=7,
            label=f"F1-Benign (ort={np.mean(f1n_vals):.3f})")
    ax.plot(folds, f1p_vals, "^-", color=C["path"], lw=2, markersize=7,
            label=f"F1-Patojenik (ort={np.mean(f1p_vals):.3f})")
    ax.set_xlabel("Fold"); ax.set_ylabel("F1 Skoru (@0.50 eşik)")
    ax.set_title("Fold × Sınıf F1 Karşılaştırması", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xticks(folds)

    # ── [2] Genel metrik karşılaştırması (grouped bar) ────────
    ax = axes[2]
    model_names = ["Ensemble", "CatBoost", "LightGBM", "XGBoost"]
    model_probs = [oof_ens, oof_cat, oof_lgb, oof_xgb]
    model_cols  = [C["ens"], C["cat"], C["lgb"], C["xgb"]]

    def safe_metrics(yp, prbs):
        try: auc = roc_auc_score(yp, prbs)
        except: auc = 0
        try: ap = average_precision_score(yp, prbs)
        except: ap = 0
        th, _ = find_best_threshold(yp, prbs)
        prd   = (prbs >= th).astype(int)
        return {
            "AUC":    round(auc, 3),
            "AP":     round(ap, 3),
            "Macro F1": round(f1_score(yp, prd, average="macro", zero_division=0), 3),
            "MCC":    round(matthews_corrcoef(yp, prd), 3),
        }

    metric_data = {n: safe_metrics(y_np, p) for n, p in zip(model_names, model_probs)}
    metric_keys = list(list(metric_data.values())[0].keys())
    x = np.arange(len(metric_keys)); width = 0.18

    for i, (name, col) in enumerate(zip(model_names, model_cols)):
        vals = [metric_data[name][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=col, alpha=0.85, edgecolor="white", lw=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_ylabel("Değer"); ax.set_ylim(0, 1.15)
    ax.set_title("Genel Metrik Karşılaştırması", fontweight="bold")
    ax.legend(fontsize=8)

    # ── [3] ROC eğrisi karşılaştırması ───────────────────────
    ax = axes[3]
    for probs, lbl, col, lw, ls in [
        (oof_ens, "Ensemble", C["ens"],  2.5, "-"),
        (oof_cat, "CatBoost", C["cat"],  1.5, "--"),
        (oof_lgb, "LightGBM", C["lgb"],  1.5, "--"),
        (oof_xgb, "XGBoost",  C["xgb"],  1.5, "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_np, probs)
        auc = roc_auc_score(y_np, probs)
        ax.plot(fpr, tpr, color=col, lw=lw, ls=ls, label=f"{lbl}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Eğrisi Karşılaştırması", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    # ── [4] PR eğrisi karşılaştırması ────────────────────────
    ax = axes[4]
    prev = y_np.mean()
    for probs, lbl, col, lw, ls in [
        (oof_ens, "Ensemble", C["ens"],  2.5, "-"),
        (oof_cat, "CatBoost", C["cat"],  1.5, "--"),
        (oof_lgb, "LightGBM", C["lgb"],  1.5, "--"),
        (oof_xgb, "XGBoost",  C["xgb"],  1.5, "--"),
    ]:
        prec, rec, _ = precision_recall_curve(y_np, probs)
        ap = average_precision_score(y_np, probs)
        ax.plot(rec, prec, color=col, lw=lw, ls=ls, label=f"{lbl}  AP={ap:.3f}")
    ax.axhline(prev, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Eğrisi Karşılaştırması", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.08)

    # ── [5] Ensemble ağırlıkları pasta ────────────────────────
    ax = axes[5]
    ax.set_facecolor(C["panel"]); ax.set_aspect("equal")
    ax.set_axis_off()
    total_w = sum(ENSEMBLE_WEIGHTS)
    pie_data = [w / total_w for w in ENSEMBLE_WEIGHTS]
    pie_labels = [f"CatBoost\n{pie_data[0]:.0%}", f"LightGBM\n{pie_data[1]:.0%}",
                  f"XGBoost\n{pie_data[2]:.0%}"]
    wedge_colors = [C["cat"], C["lgb"], C["xgb"]]
    wedges, texts = ax.pie(pie_data, labels=pie_labels, colors=wedge_colors,
                           startangle=90, wedgeprops=dict(edgecolor="white", lw=2),
                           textprops=dict(fontsize=11, fontweight="bold"))
    ax.set_title("Ensemble Ağırlık Dağılımı", fontweight="bold", pad=10)

    fig.suptitle(f"PANEL: {panel}  —  Model ve Fold Karşılaştırma Analizi",
                 fontsize=14, fontweight="bold", y=1.01, color=C["text"])

    path = save_dir / f"{panel}_4_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [4/5] Model karşılaştırma kaydedildi.")


# ─────────────────────────────────────────────────────────────────
# Figure 5 — Hiper-Parametre Profili
# ─────────────────────────────────────────────────────────────────
def fig5_hyperparams(X, y, sensitivity_results, panel, save_dir):
    depths = sorted(set(k[0] for k in sensitivity_results))
    lrs    = sorted(set(k[1] for k in sensitivity_results))
    heatmap_data = np.array([[sensitivity_results.get((d, lr), np.nan)
                               for lr in lrs] for d in depths])

    fig = plt.figure(figsize=(18, 11), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor(C["panel"])

    # ── [0] Parametre tablosu görselleştirmesi ────────────────
    ax = axes[0]; ax.set_axis_off()
    params_table = [
        ["Parametre",        "CatBoost",      "LightGBM",     "XGBoost"],
        ["iterations/trees", "600",           "400",          "400"],
        ["learning_rate",    "0.03",          "0.04",         "0.04"],
        ["depth/max_depth",  "4",             "4",            "4"],
        ["l2 / reg_lambda",  "5  (l2_leaf)",  "1.0",          "2.0"],
        ["l1 / reg_alpha",   "—",             "0.10",         "0.10"],
        ["min_samples",      "—",             "12 (child)",   "5.0 (weight)"],
        ["subsample",        "—",             "0.80",         "0.80"],
        ["colsample",        "—",             "0.80",         "0.80"],
        ["gamma",            "—",             "—",            "1.0"],
        ["class_weight",     "Balanced (auto)","scale_pos_wt", "scale_pos_wt"],
        ["Metrik",           "AUC (eval)",    "binary loss",  "logloss"],
    ]

    col_widths = [0.30, 0.23, 0.23, 0.23]
    row_h = 0.071
    header_cols = [C["accent"], C["cat"], C["lgb"], C["xgb"]]

    for ri, row in enumerate(params_table):
        for ci, cell in enumerate(row):
            x0 = sum(col_widths[:ci])
            y0 = 1.0 - (ri + 1) * row_h
            bg = header_cols[ci] if ri == 0 else ("#eaf4fb" if ri % 2 == 1 else C["bg"])
            fc = ax.add_patch(mpatches.FancyBboxPatch(
                (x0 + 0.005, y0 + 0.003), col_widths[ci] - 0.010, row_h - 0.006,
                boxstyle="round,pad=0.005", facecolor=bg, edgecolor=C["grid"], lw=0.5,
                transform=ax.transAxes))
            tc = "white" if ri == 0 else C["text"]
            fw = "bold" if ri == 0 or ci == 0 else "normal"
            ax.text(x0 + col_widths[ci] / 2, y0 + row_h / 2, cell,
                    ha="center", va="center", fontsize=8.5,
                    color=tc, fontweight=fw, transform=ax.transAxes)

    ax.set_title("Hiper-Parametre Profili  (3 Model Karşılaştırması)",
                 fontweight="bold", pad=10)

    # ── [1] Duyarlılık ısı haritası (depth × LR) ─────────────
    ax = axes[1]
    sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                xticklabels=[f"{lr}" for lr in lrs],
                yticklabels=[str(d) for d in depths],
                linewidths=1.0, linecolor="white",
                cbar_kws={"label": "OOF MCC (3-fold)", "shrink": 0.75},
                vmin=0, vmax=1)

    # Seçilen değerleri işaretle
    best_d, best_lr = max(sensitivity_results, key=lambda k: sensitivity_results[k])
    di = depths.index(best_d); li = lrs.index(best_lr)
    ax.add_patch(mpatches.Rectangle((li, di), 1, 1, fill=False,
                                    edgecolor=C["thresh"], lw=3))
    ax.text(li + 0.5, di + 0.5, "★", ha="center", va="center",
            fontsize=18, color=C["thresh"])

    ax.set_xlabel("Öğrenme Oranı (learning_rate)", fontsize=10)
    ax.set_ylabel("Ağaç Derinliği (depth)", fontsize=10)
    ax.set_title(
        f"CatBoost Duyarlılık Haritası\n"
        f"(En iyi: depth={best_d}, lr={best_lr}  →  MCC={sensitivity_results[(best_d,best_lr)]:.3f})",
        fontweight="bold", pad=8)

    # ── [2] Balanced bagging görselleştirmesi ─────────────────
    ax = axes[2]; ax.set_facecolor(C["panel"])
    n_neg  = int((y == 0).sum()); n_pos = int(y.sum())
    bag_neg = n_neg
    bag_pos = min(int(np.ceil(n_neg * PATHOGENIC_TO_BENIGN_RATIO)), n_pos)
    categories = ["Tüm Veri\n(ham)", "Bag başına\n(dengeli)"]
    benign_vals = [n_neg, bag_neg]
    path_vals   = [n_pos, bag_pos]
    x = np.array([0, 1])
    ax.bar(x - 0.2, benign_vals, 0.35, color=C["benign"], alpha=0.85,
           label=f"Benign (0)  [n={n_neg}]", edgecolor="white")
    ax.bar(x + 0.2, path_vals,   0.35, color=C["path"],   alpha=0.85,
           label=f"Patojenik (1)  [n={n_pos}]", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Örnek Sayısı")
    ax.set_title(
        f"Balanced Bagging Strateji\n"
        f"({BALANCED_BAGS} bag/fold  |  ratio={PATHOGENIC_TO_BENIGN_RATIO:.1f}×)",
        fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    for xi, (bv, pv) in enumerate(zip(benign_vals, path_vals)):
        ax.text(xi - 0.2, bv + 0.5, str(bv), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C["benign"])
        ax.text(xi + 0.2, pv + 0.5, str(pv), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C["path"])
    ratio_before = n_pos / max(n_neg, 1)
    ratio_after  = bag_pos / max(bag_neg, 1)
    ax.text(0.98, 0.98, f"Oran  →  {ratio_before:.1f}:1  →  {ratio_after:.1f}:1",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=C["text"], style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe5e5", lw=0.5))
    ax.grid(axis="y", color=C["grid"], lw=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── [3] Düzenlileştirme etkisi çubukları ─────────────────
    ax = axes[3]; ax.set_facecolor(C["panel"])
    reg_params = {
        "CatBoost\nl2_leaf_reg": 5 / 10,
        "CatBoost\ndepth": 4 / 8,
        "LightGBM\nreg_alpha": 0.10 / 4.0,
        "LightGBM\nreg_lambda": 1.0 / 10.0,
        "XGBoost\ngamma": 1.0 / 4.0,
        "XGBoost\nreg_alpha": 0.10 / 4.0,
        "XGBoost\nreg_lambda": 2.0 / 10.0,
        "LightGBM\nmin_child": 12 / 30,
    }
    rp_names = list(reg_params.keys())
    rp_vals  = list(reg_params.values())
    rp_cols  = [C["cat"] if "CatBoost" in n else
                (C["lgb"] if "LightGBM" in n else C["xgb"]) for n in rp_names]
    ax.barh(rp_names, rp_vals, color=rp_cols, alpha=0.85,
            edgecolor="white", height=0.65)
    ax.axvline(0.5, color=C["thresh"], lw=1.5, ls="--", label="Orta değer (normalize)")
    ax.set_xlabel("Normalize düzenlileştirme gücü  [0=zayıf, 1=güçlü]")
    ax.set_title("Düzenlileştirme Parametreleri Profili\n(Overfitting Önleme)",
                 fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color=C["grid"], lw=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for i, v in enumerate(rp_vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=C["text"])

    fig.suptitle(f"PANEL: {panel}  —  Hiper-Parametre Seçim Analizi",
                 fontsize=14, fontweight="bold", y=1.01, color=C["text"])

    path = save_dir / f"{panel}_5_hyperparams.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  [5/5] Hiper-parametre analizi kaydedildi.")


# ─────────────────────────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────────────────────────
def main(panel: str):
    panel = panel.upper()
    if panel not in PANEL_CSV:
        print(f"Geçersiz panel: {panel}. Seçenekler: {list(PANEL_CSV.keys())}")
        sys.exit(1)

    save_dir = OUTPUTS_DIR / "figures" / panel
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  PANEL: {panel}  —  Kapsamlı Analiz Başlıyor")
    print(f"  Çıktı klasörü: {save_dir}")
    print(f"{'=' * 60}")

    # ── Veri yükle ────────────────────────────────────────────
    print("\n[1] Veri yükleniyor...")
    feat_df, raw_p = load_features(panel)
    X = feat_df.drop(columns=[TARGET], errors="ignore")
    y = feat_df[TARGET]
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    print(f"  Satir={len(y)}  POS={n_pos}  NEG={n_neg}  Ozellik={X.shape[1]}")

    raw_feat    = raw_p.drop(columns=["Variant_ID", TARGET, "panel"], errors="ignore")
    groups      = make_exact_row_groups(raw_feat)
    dup_weights = inverse_duplicate_weights(groups)

    # ── OOF çalıştır ──────────────────────────────────────────
    print(f"\n[2] OOF değerlendirme ({N_FOLDS}-fold, {BALANCED_BAGS} bag)...")
    oof_ens, oof_cat, oof_lgb, oof_xgb, fold_rows = run_oof(
        X, y, groups, dup_weights, verbose=True)

    threshold, _ = find_best_threshold(y.values, oof_ens)
    print(f"  Seçilen eşik: {threshold:.4f}")

    # ── Kaydedilmiş modeli yükle (SHAP için) ──────────────────
    model_path = OUTPUTS_DIR / f"model_{panel}.pkl"
    final_cb = None
    if model_path.exists():
        print(f"\n[3] Kaydedilmiş model yükleniyor: {model_path.name}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        final_cb = bundle["models"][0]
    else:
        print(f"\n[3] Model dosyası bulunamadı. SHAP için tüm veri üzerinde CatBoost eğitiliyor...")
        final_cb = CatBoostClassifier(**CAT_PARAMS)
        final_cb.fit(X, y, sample_weight=dup_weights)

    # ── Hiper-parametre duyarlılık analizi ────────────────────
    print(f"\n[4] Hiper-parametre duyarlılık analizi (3-fold, CatBoost)...")
    sensitivity = run_sensitivity(X, y)

    # ── Grafikler ─────────────────────────────────────────────
    print(f"\n[5] Grafikler oluşturuluyor...")
    metrics = fig1_performance(y, oof_ens, oof_cat, oof_lgb, oof_xgb,
                                threshold, panel, save_dir)
    fig2_shap(final_cb, X, panel, save_dir)
    fig4_comparison(y, oof_ens, oof_cat, oof_lgb, oof_xgb, fold_rows, panel, save_dir)
    fig5_hyperparams(X, y, sensitivity, panel, save_dir)

    # ── Sonuç özeti ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PANEL: {panel}  —  SONUÇ ÖZETİ")
    print(f"{'=' * 60}")
    print(f"  Macro F1        : {metrics['f1']:.4f}")
    print(f"  MCC             : {metrics['mcc']:.4f}")
    print(f"  PR-AUC (AP)     : {metrics['ap']:.4f}")
    print(f"  ROC-AUC         : {metrics['auc']:.4f}")
    print(f"  Patojenik Recall: {metrics['path_recall']:.4f}")
    print(f"  Benign Recall   : {metrics['benign_recall']:.4f}")
    print(f"  Eşik            : {metrics['threshold']:.4f}")
    print(f"\n  Kaydedilen dosyalar ({save_dir}):")
    for f in sorted(save_dir.glob(f"{panel}_*.png")):
        print(f"    {f.name}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python src/analyze_panel.py <PANEL>")
        print("Ornek   : python src/analyze_panel.py PAH")
        sys.exit(1)
    main(sys.argv[1])
