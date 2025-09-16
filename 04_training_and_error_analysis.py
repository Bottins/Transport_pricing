# -*- coding: utf-8 -*-
"""
04_training_and_error_analysis.py — FIX:
1) Salva anche GMM per ciascun tipo (models/gmm_{Tipo}.joblib) + metadata con scale_method e n_components.
2) Salva su Excel UNA SOLA VOLTA alla fine (tutti i fogli in un unico writer).
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

from sklearn.mixture import GaussianMixture  # GMM

# ==== Parametri incertezza / GMM ====
SCALE_METHOD = "gmm_global"   # ["gmm_global", "gmm_local", "trimmed_std"]
MAX_GMM_COMPONENTS = 4
TRIM_Q = 0.10

# ======================
# Parametri generali
# ======================

ID_COL = "idordine"
QUOTE_COL = "idquotazione"
TIPO_COL = "tipo_carico"
TARGET_COL = "prezzo_attualizzato"

EXCLUDE_COLUMNS: List[str] = [
    "idquotazione","idordine", TARGET_COL,
    "prezzo_carb","longitudine_scarico","importo_per_peso",
    "importo","importotrasp","importo_per_km","importo_norm","Coefficiente",
    "stato_quotazione","estimated","ordine_originale"
]

N_ESTIMATORS = 750
RANDOM_STATE = 1
TEST_SIZE = 0.1
INTERVAL_PERCENT = 50.0

VERBOSE = True
DEBUG_PLOTS = True
SAVE_PLOTS = True
PLOTS_DPI = 200

APE_MAX = 100.0
APE_BIN = 1.0
EXTRA_RESULT_COLS = ["ordine_originale","tipo_carico","reg_carico","reg_scarico"]

# ======================
# Cartelle output
# ======================
FIGDIR = "figs"
MODELDIR = "models"
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

def _log(msg: str):
    if VERBOSE:
        print(msg, file=sys.stdout)

def _savefig(filename: str):
    if SAVE_PLOTS:
        path = os.path.join(FIGDIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=PLOTS_DPI)
    plt.close()

def check_invariance_within_order(df: pd.DataFrame,
                                  id_col: str,
                                  target_col: str,
                                  quote_col: Optional[str] = None,
                                  log_warnings: bool = False):
    cols_to_check = [c for c in df.columns if c not in [id_col, target_col] + ([quote_col] if quote_col else [])]
    bad_cols = []
    for c in cols_to_check:
        nun = df.groupby(id_col)[c].nunique(dropna=False)
        if (nun > 1).any():
            bad_cols.append(c)
    if log_warnings and bad_cols:
        _log(f"[WARN] Colonne non costanti entro ordine: {bad_cols}")
    return bad_cols

def collapse_orders_median_target(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    quote_col: Optional[str] = None,
    prefer: str = "upper",
    set_target_to_median: bool = True
) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy()
    out["_idx"] = np.arange(len(out))
    out["_med"] = out.groupby(id_col)[target_col].transform("median")
    out["_dist"] = (out[target_col] - out["_med"]).abs()
    if prefer == "lower":
        out["_bias"] = np.where(out[target_col] <= out["_med"], 0, 1)
    elif prefer == "upper":
        out["_bias"] = np.where(out[target_col] >= out["_med"], 0, 1)
    else:
        out["_bias"] = 0
    sort_cols, ascending = [id_col, "_dist", "_bias"], [True, True, True]
    if quote_col is not None and quote_col in out.columns:
        sort_cols.append(quote_col); ascending.append(True)
    sort_cols.append("_idx"); ascending.append(True)
    picked = (out.sort_values(sort_cols, ascending=ascending)
                 .drop_duplicates(subset=[id_col], keep="first").copy())
    if set_target_to_median:
        med = out.groupby(id_col, as_index=False)["_med"].first()
        picked = picked.drop(columns=[target_col], errors="ignore").merge(
            med.rename(columns={"_med": target_col}), on=id_col, how="left"
        )
    picked = picked.drop(columns=[c for c in ["_idx","_med","_dist","_bias"] if c in picked.columns])
    if quote_col is not None and quote_col in picked.columns:
        picked = picked.drop(columns=[quote_col])
    return picked.reset_index(drop=True)

def build_pipeline(X: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features),
                     ("cat", categorical_transformer, categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=RANDOM_STATE, criterion="absolute_error"
    )
    return Pipeline(steps=[("prep", preprocessor), ("rf", model)]), numeric_features, categorical_features

def get_feature_importances(pipe: Pipeline) -> pd.DataFrame:
    preproc = pipe.named_steps["prep"]; rf: RandomForestRegressor = pipe.named_steps["rf"]
    try:
        names = preproc.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(len(rf.feature_importances_))], dtype=object)
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    return imp.reset_index(drop=True)

def show_feature_importances(pipe: Pipeline, tipo: str, top_n: int = 10):
    df_imp = get_feature_importances(pipe)
    _log(f"\n=== Feature importances — {tipo} (top {top_n}) ===")
    _log(df_imp.head(top_n).to_string(index=False))
    top = df_imp.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, max(4, 0.35*len(top))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance"); plt.title(f"RF Feature Importances — {tipo} (top {top_n})")
    _savefig(f"feature_importances_{tipo}.png")

def per_prediction_uncertainty(model: RandomForestRegressor, X_tx: np.ndarray, interval_percent: float):
    lower_q = (100 - interval_percent) / 2
    upper_q = 100 - lower_q
    tree_preds = np.stack([est.predict(X_tx) for est in model.estimators_], axis=1)
    mean_pred = np.mean(tree_preds, axis=1)
    std_pred = np.std(tree_preds, axis=1, ddof=1)
    lower = np.percentile(tree_preds, lower_q, axis=1)
    upper = np.percentile(tree_preds, upper_q, axis=1)
    return mean_pred, std_pred, np.stack([lower, upper], axis=1)

# ===== Incertezza (GMM helpers) =====
def _gmm_fit_on_target(y: np.ndarray, max_components: int = 4, random_state: int = 0) -> GaussianMixture:
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    models, bics = [], []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gm.fit(y)
        models.append(gm); bics.append(gm.bic(y))
    return models[int(np.argmin(bics))]

def _gmm_mixture_std(gm: GaussianMixture) -> float:
    w = gm.weights_
    mu = gm.means_.flatten()
    sigma2 = np.array([np.squeeze(gm.covariances_[i]) for i in range(gm.n_components)], dtype=float)
    mu_mix = np.sum(w * mu)
    var_mix = np.sum(w * (sigma2 + (mu - mu_mix) ** 2))
    return float(np.sqrt(max(var_mix, 1e-12)))

def _gmm_local_std(gm: GaussianMixture, y_hat: float) -> float:
    r = gm.predict_proba(np.array([[float(y_hat)]]))[0]
    k = int(np.argmax(r))
    sigma2_k = float(np.squeeze(gm.covariances_[k]))
    return float(np.sqrt(max(sigma2_k, 1e-12)))

def _trimmed_std(y: np.ndarray, q: float = 0.10) -> float:
    y = np.asarray(y, dtype=float)
    lo, hi = np.quantile(y, [q, 1 - q])
    yt = y[(y >= lo) & (y <= hi)]
    if yt.size < 2: yt = y
    return float(np.std(yt, ddof=1) if yt.size > 1 else 0.0)

def _confidence_from_scale(std_pred: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    eps = 1e-9
    denom = np.asarray(std_pred, dtype=float) + np.asarray(scale, dtype=float) + eps
    conf = 1.0 - (std_pred / denom)
    return np.clip(conf, 0.0, 1.0)

# ===== Salvataggio bundle (RF + features + metadata + GMM opzionale) =====
def _save_model_bundle(tipo: str, pipe: Pipeline, feature_cols: List[str], meta: dict, gmm: GaussianMixture | None = None):
    model_path = os.path.join(MODELDIR, f"rf_model_{tipo}.joblib")
    joblib.dump(pipe, model_path)
    features_path = os.path.join(MODELDIR, f"features_{tipo}.txt")
    with open(features_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    meta_path = os.path.join(MODELDIR, f"metadata_{tipo}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if gmm is not None:
        gmm_path = os.path.join(MODELDIR, f"gmm_{tipo}.joblib")
        joblib.dump(gmm, gmm_path)
        _log(f"[MODEL] Salvato GMM: {gmm_path}")
    _log(f"[MODEL] Salvati: {model_path}, {features_path}, {meta_path}")

def fit_and_evaluate_by_tipo(df: pd.DataFrame, tipo_col: str, target_col: str, id_col: str,
                             exclude_cols: List[str], interval_percent: float):
    results_summary = {}
    predictions_per_tipo = {}
    base_exclude = set(exclude_cols) | {id_col, tipo_col, target_col}

    for tipo_value in ["Completo","Parziale","Groupage"]:
        subset = df[df[tipo_col] == tipo_value].copy()
        _log(f"\n[MODEL] Tipo '{tipo_value}': righe = {len(subset)}")
        if subset.empty or subset[target_col].notna().sum() == 0:
            _log(f"[MODEL] Skip '{tipo_value}' (vuoto o target NaN)")
            continue

        subset["mese_ordine"] = pd.to_datetime(subset["data_ordine"]).dt.month
        if tipo_value == "Groupage":
            extra_drop = {"data_ordine","spazio_calcolato"}
        else:
            extra_drop = {"data_ordine", "altezza", "lunghezza_max","spazio_calcolato"}

        feature_cols = [c for c in subset.columns if c not in (base_exclude | extra_drop)]
        X = subset[feature_cols].copy()
        y = subset[target_col].astype(float).copy()
        id_values = subset[id_col].values

        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, id_values, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        pipe, _, _ = build_pipeline(X_train)
        pipe.fit(X_train, y_train)

        show_feature_importances(pipe, tipo_value, top_n=10)

        # Metriche TRAIN
        y_pred_train = pipe.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mape_train_pct = float(mean_absolute_percentage_error(y_train, y_pred_train) * 100.0)
        _log(f"[METRICS][TRAIN] {tipo_value} | MAE={mae_train:.4f} | MAPE={mape_train_pct:.2f}%")

        # Metriche TEST
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape_pct = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)
        _log(f"[METRICS][TEST] {tipo_value} | MAE={mae:.4f} | MAPE={mape_pct:.2f}%")

        # Incertezza per-predizione
        preproc = pipe.named_steps["prep"]; rf: RandomForestRegressor = pipe.named_steps["rf"]
        X_test_tx = preproc.transform(X_test)
        mean_pred, std_pred, intervals = per_prediction_uncertainty(rf, X_test_tx, interval_percent)

        # === Scala "consapevole della distribuzione" (GMM) ===
        gmm = None
        meta_scale = {"scale_method": SCALE_METHOD}

        if SCALE_METHOD == "gmm_global":
            gmm = _gmm_fit_on_target(y_train, max_components=MAX_GMM_COMPONENTS, random_state=RANDOM_STATE)
            scale_global = _gmm_mixture_std(gmm)
            confidence = _confidence_from_scale(std_pred, scale_global)
        elif SCALE_METHOD == "gmm_local":
            gmm = _gmm_fit_on_target(y_train, max_components=MAX_GMM_COMPONENTS, random_state=RANDOM_STATE)
            local_scales = np.array([_gmm_local_std(gmm, yh) for yh in mean_pred], dtype=float)
            confidence = _confidence_from_scale(std_pred, local_scales)
        elif SCALE_METHOD == "trimmed_std":
            scale_trim = _trimmed_std(y_train.values, q=TRIM_Q)
            meta_scale["trim_q"] = TRIM_Q
            confidence = _confidence_from_scale(std_pred, scale_trim)
        else:
            mad_target = np.median(np.abs(y_train - np.median(y_train))) + 1e-9
            confidence = 1.0 - (std_pred / (std_pred + mad_target))

        confidence = confidence.astype(float)

        # Output per fogli di test
        eps = 1e-9
        ape_percent = np.abs((y_test.values - mean_pred) / (np.abs(y_test.values) + eps)) * 100.0
        out_df = pd.DataFrame({
            id_col: id_test,
            "y_true": y_test.values,
            "y_pred": mean_pred,
            "pred_std": std_pred,
            "conf_approx": confidence,
            f"pi{interval_percent}_lower": intervals[:,0],
            f"pi{interval_percent}_upper": intervals[:,1],
            "ape_percent": ape_percent,
        })
        for col in EXTRA_RESULT_COLS:
            if col in subset.columns:
                mapping = dict(zip(id_values, subset[col]))
                out_df[col] = [mapping.get(i, np.nan) for i in id_test]

        predictions_per_tipo[tipo_value] = out_df
        results_summary[tipo_value] = {
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "MAE": float(mae), "MAPE_percent": float(mape_pct)
        }

        # === Salvataggio MODEL BUNDLE (RF + features + metadata + GMM) ===
        meta = {
            "tipo": tipo_value,
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "test_size": TEST_SIZE,
            "interval_percent": interval_percent,
            "n_train": int(len(X_train)),
            "scale_method": SCALE_METHOD,
            "gmm_components": (int(gmm.n_components) if gmm is not None else None)
        }
        _save_model_bundle(tipo_value, pipe, feature_cols, meta, gmm=gmm)

        if DEBUG_PLOTS:
            plt.figure(); plt.scatter(y_test.values, mean_pred, alpha=0.6)
            plt.xlabel("y_true"); plt.ylabel("y_pred"); plt.title(f"y_true vs y_pred — {tipo_value}")
            _savefig(f"scatter_true_vs_pred_{tipo_value}.png")

            resid = y_test.values - mean_pred
            plt.figure(); plt.hist(resid, bins=30)
            plt.xlabel("Residuo (y_true - y_pred)"); plt.ylabel("Frequenza"); plt.title(f"Residui — {tipo_value}")
            _savefig(f"residuals_{tipo_value}.png")

    return results_summary, predictions_per_tipo

def plot_ape_distributions(preds: Dict[str, pd.DataFrame]):
    for tipo, df_pred in preds.items():
        if "ape_percent" not in df_pred.columns: continue
        vals = df_pred["ape_percent"].dropna().values
        vals = vals[(vals >= 0) & (vals <= APE_MAX)]
        if vals.size == 0: continue
        plt.figure()
        bins = np.arange(0, APE_MAX + APE_BIN, APE_BIN)
        plt.hist(vals, bins=bins)
        plt.title(f"Distribuzione APE% - {tipo}")
        plt.xlabel("APE (%)"); plt.ylabel("Frequenza")
        _savefig(f"ape_distribution_{tipo}.png")

def plot_error_vs_conf_by_tipo(preds: Dict[str, pd.DataFrame], bin_width: float = 0.10, use_ape: bool = True):
    for tipo, dfp in preds.items():
        if "conf_approx" not in dfp.columns: continue
        conf = dfp["conf_approx"].astype(float).values
        if use_ape and "ape_percent" in dfp.columns:
            err = dfp["ape_percent"].astype(float).values; ylabel = "APE (%)"
        else:
            err = np.abs(dfp["y_true"].astype(float).values - dfp["y_pred"].astype(float).values); ylabel = "Errore assoluto"
        bins = np.arange(0.0, 1.0 + 1e-9, bin_width)
        data, labels = [], []
        for i in range(len(bins) - 1):
            b0, b1 = bins[i], bins[i+1]
            mask = (conf >= b0) & (conf < b1) if i < len(bins)-2 else (conf >= b0) & (conf <= b1)
            e = err[mask]
            if e.size > 0:
                data.append(e); labels.append(f"{b0:.2f}-{b1:.2f}\n(n={e.size})")
        if not data: continue
        plt.figure(figsize=(max(8, len(data)*0.6), 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right"); plt.ylabel(ylabel)
        plt.title(f"Errori vs Confidenza (bin={bin_width:.2f}) — {tipo}")
        _savefig(f"errori_vs_conf_{tipo}.png")


def recalibrate_target_by_order(
    df: pd.DataFrame,
    id_col: str = "idordine",
    importo_norm_col: str = "importo_norm",
    km_col: str = "km_tratta",
    spazio_col: str = "spazio_calcolato",
    target_col: str = "prezzo_attualizzato",
) -> pd.DataFrame:
    """
    Per ogni idordine:
      - calcola la mediana di importo_norm
      - sostituisce prezzo_attualizzato con mediana_importo_norm * km_tratta * spazio_calcolato / 10000
    Mantiene TUTTE le righe (nessun collasso).
    """
    df = df.copy()
    if any(c not in df.columns for c in [id_col, importo_norm_col, km_col, spazio_col]):
        missing = [c for c in [id_col, importo_norm_col, km_col, spazio_col] if c not in df.columns]
        raise KeyError(f"Mancano colonne richieste per la ricalibrazione: {missing}")

    # mediana per ordine
    med_per_order = (
        df.groupby(id_col, dropna=False)[importo_norm_col]
          .median()
          .rename("_med_importo_norm")
    )
    df = df.merge(med_per_order, left_on=id_col, right_index=True, how="left")

    # nuova definizione del target
    df[target_col] = (
        df["_med_importo_norm"].astype(float)
        * df[km_col].astype(float)
        * df[spazio_col].astype(float)
        / 1e5
    )

    df.drop(columns=["_med_importo_norm"], inplace=True)
    return df

# ======================
# Esecuzione
# ======================
EXCEL_FILE = "03_risultati_matchati.xlsx"
INPUT_SHEET = "Matched"

print(f"[LOAD] Leggo: {EXCEL_FILE}, foglio '{INPUT_SHEET}'")
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=INPUT_SHEET)

# Check (facoltativo)
check_invariance_within_order(df_raw, id_col=ID_COL, target_col=TARGET_COL, quote_col=QUOTE_COL, log_warnings=False)

# Collassa a una riga per ordine
df_grouped = collapse_orders_median_target(df_raw, id_col=ID_COL, target_col=TARGET_COL, quote_col=QUOTE_COL)


# Ricalibra il target mantenendo TUTTE le righe (nessun collasso)
df_calibrated = recalibrate_target_by_order(
    df_raw,
    id_col=ID_COL,
    importo_norm_col="importo_norm",
    km_col="km_tratta",
    spazio_col="spazio_calcolato",
    target_col=TARGET_COL,
)
# --- RACCOGLITORE FOGLI (scriveremo tutto alla fine) ---
excel_sheets: dict[str, pd.DataFrame] = {}

# Foglio 'Aggregated'
DROP_COLS_INTERMEDIO = ["importo","importotrasp","stato_quotazione","ordine_originale"]
# df_to_save = df_grouped.drop(columns=DROP_COLS_INTERMEDIO, errors="ignore")
df_to_save = df_calibrated.drop(columns=DROP_COLS_INTERMEDIO, errors="ignore")
excel_sheets["Aggregated"] = df_to_save.copy()
_log(f"[OK] Pronto foglio 'Aggregated' per export unico finale")

# Training + valutazione
summary_metrics, preds = fit_and_evaluate_by_tipo(
    df_to_save, tipo_col=TIPO_COL, target_col=TARGET_COL, id_col=ID_COL,
    exclude_cols=EXCLUDE_COLUMNS, interval_percent=INTERVAL_PERCENT
)

plot_ape_distributions(preds)
plot_error_vs_conf_by_tipo(preds, bin_width=0.10, use_ape=True)


# Estrazione peggiori per APE
WORST_K = 1000
base_exclude = set(EXCLUDE_COLUMNS) | {ID_COL, TIPO_COL, TARGET_COL}
all_features = [c for c in df_to_save.columns if c not in base_exclude]

worst_by_tipo = {}
for tipo in ["Completo","Parziale","Groupage"]:
    if tipo not in preds: 
        continue
    df_pred = preds[tipo].copy()
    if "ape_percent" not in df_pred.columns or df_pred.empty:
        continue
    feat_subset = df_to_save[df_to_save[TIPO_COL] == tipo][[ID_COL] + all_features].copy()
    df_join = df_pred.merge(feat_subset, on=ID_COL, how="left")
    cols_metriche = [ID_COL,"y_true","y_pred","ape_percent","pred_std","conf_approx"] + [c for c in df_pred.columns if c.startswith("pi")]
    ordered_cols = [c for c in cols_metriche if c in df_join.columns] + [c for c in all_features if c in df_join.columns]
    worst_by_tipo[tipo] = df_join.sort_values("ape_percent", ascending=False).head(WORST_K)[ordered_cols]
    excel_sheets[f"Test_{tipo}"[:31]] = worst_by_tipo[tipo].copy()
_log("[OK] Pronti fogli Test_* per export unico finale")

# === Riepilogo MAPE ===
# 1) MAPE previsioni: direttamente dai DataFrame in memoria (non leggiamo da Excel)
mape_previsioni = {}
for tipo, categoria in [("Completo","completo"), ("Parziale","parziale"), ("Groupage","groupage")]:
    if tipo in preds and not preds[tipo].empty and "ape_percent" in preds[tipo].columns:
        mape_previsioni[categoria] = float(np.nanmean(preds[tipo]["ape_percent"].values))
    else:
        mape_previsioni[categoria] = float("nan")

# 2) MAPE di 'estimated' dai dati o02_risultariginali (leggiamo solo questo foglio dal file esistente)
FILTRATI_FILE="02_risultati_filtrati.xlsx"
df_f = pd.read_excel(FILTRATI_FILE, sheet_name="Risultati_filtrati")
df_f = df_f[df_f["importotrasp"] > 0].copy()
df_f["ape_estimated"] = (
    (df_f["importotrasp"] - df_f["estimated"]).abs()
    / df_f["importotrasp"] * 100.0
)
mape_estimated = (
    df_f.groupby(df_f["tipo_carico"].str.lower())["ape_estimated"]
        .mean()
        .reindex(["completo", "parziale", "groupage"])
        .to_dict()
)

summary = pd.DataFrame({
    "categoria": ["completo", "parziale", "groupage"],
    "MAPE_previsioni_%": [
        round(mape_previsioni.get("completo", float("nan")), 3),
        round(mape_previsioni.get("parziale", float("nan")), 3),
        round(mape_previsioni.get("groupage", float("nan")), 3),
    ],
    "MAPE_estimated_%": [
        round(mape_estimated.get("completo", float("nan")), 3),
        round(mape_estimated.get("parziale", float("nan")), 3),
        round(mape_estimated.get("groupage", float("nan")), 3),
    ],
})
summary["Miglioramento_%"] = (
    (summary["MAPE_estimated_%"] - summary["MAPE_previsioni_%"])
    / summary["MAPE_estimated_%"] * 100
).round(2)

excel_sheets["MAPE_categorie"] = summary.copy()
_log("[OK] Pronto foglio 'MAPE_categorie' per export unico finale")

# === SCRITTURA UNICA 
output_file="04_risultati.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
    for sheet_name, df_sheet in excel_sheets.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
print(f"[OK] Tutti i fogli salvati in '{output_file}'")
