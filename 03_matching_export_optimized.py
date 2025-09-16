# -*- coding: utf-8 -*-
"""
03_matching_export_optimized.py
Velocizzato e con parallelizzazione opzionale:
- Matching singleton -> base completamente vettorializzato per gruppi di colonne uguali.
- Clustering tra singleton basato su "bucketing" (discretizzazione per tolleranze) O(n) per gruppo,
  evitando doppi cicli annidati O(n^2).
- (Opzionale) parallelizzazione per chiave di gruppo (uguali EQUAL_COLS) usando ProcessPoolExecutor.
  NOTA: per la parallelizzazione è necessario il guard __main__.

Compatibile con l'output originale: preserva 'ordine_originale' e salva 'matched.xlsx'.
Niente seaborn; grafici con solo matplotlib (più leggero).
"""

import os
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# ======================
# Parametri / Percorsi
# ======================
ID_COL = "idordine"
QUOTE_COL = "idquotazione"

# Colonne per la regola di matching
EQUAL_COLS = ["tipo_pallet","tipo_carico","is_isola","estero"]

# - ('abs', valore)  -> tolleranza assoluta (come prima)
# - ('rel', frazione)-> tolleranza percentuale (es. 0.10 = ±10%)
TOL_NUMERIC = {
    "Perc_camion": ("abs", 0.05),
    "Tassativi": ("abs", 0.05),
    "km_tratta": ("rel", 0.10),       # ±10%
    "verso_nord": ("rel", 0.30),      
}


# Parallelismo (per default OFF — attivalo se hai molte chiavi e CPU libera)
PARALLEL = True
MAX_WORKERS = max(1, os.cpu_count() or 1)

# ======================
# Helpers
# ======================
@dataclass
class MatchRule:
    equal_cols: list
    tol_numeric: dict

def _build_representatives(df: pd.DataFrame, id_col: str, cols: list) -> pd.DataFrame:
    """Rappresentanti per idordine: mediane per numeriche, primo valido per categoriche."""
    num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c in df.columns and c not in num_cols]
    agg = {c: "median" for c in num_cols}
    for c in cat_cols:
        agg[c] = (lambda x: x.dropna().iloc[0] if x.dropna().shape[0] > 0 else np.nan)
    reps = df.groupby(id_col, as_index=False).agg(agg)
    return reps

def _vectorized_best_match(singleton_block: pd.DataFrame, base_block: pd.DataFrame,
                           tol_cols: list, tol_kinds: list, tol_vals: np.ndarray, id_col: str) -> dict:
    id_map = {}
    if base_block.empty or singleton_block.empty or not tol_cols:
        return id_map

    B = base_block[tol_cols].to_numpy(dtype=float, copy=False)
    base_ids = base_block[id_col].to_numpy(copy=False)
    eps = 1e-12

    for _, r in singleton_block.iterrows():
        s = r[tol_cols].to_numpy(dtype=float, copy=False)
        if np.isnan(s).any():
            continue

        # tolleranza per colonna riferita al valore del singleton (per 'rel')
        tol_row = np.empty_like(s, dtype=float)
        for j, (kind, val) in enumerate(zip(tol_kinds, tol_vals)):
            if kind == "abs":
                tol_row[j] = max(val, eps)
            else:  # "rel"
                tol_row[j] = max(val * abs(s[j]), eps)

        diffs = np.abs(B - s)
        within = (diffs <= tol_row).all(axis=1)
        if not within.any():
            continue

        scores = (diffs[within] / tol_row).sum(axis=1)
        best_pos = np.argmin(scores)
        id_map[int(r[id_col])] = int(base_ids[within][best_pos])
    return id_map



def _bucket_key(row_vals: np.ndarray, tol_kinds: list, tol_vals: np.ndarray) -> tuple:
    if np.isnan(row_vals).any():
        return ("__nan__",) + tuple(np.isnan(row_vals))

    key_parts = []
    eps = 1e-12
    for v, kind, val in zip(row_vals, tol_kinds, tol_vals):
        if kind == "abs":
            denom = val if val > 0 else eps
            key_parts.append(("abs", math.floor(v / denom)))
        else:
            # bucket logaritmico per tolleranza relativa (rapporto)
            sign = 1 if v >= 0 else -1
            av = max(abs(v), eps)
            step = math.log1p(val) if val > 0 else eps  # log(1+p)
            key_parts.append(("rel", sign, math.floor(math.log(av) / step)))
    return tuple(key_parts)

def _cluster_singletons(singleton_block: pd.DataFrame, tol_cols: list,
                        tol_kinds: list, tol_vals: np.ndarray, id_col: str) -> dict:
    id_map = {}
    if singleton_block.empty or not tol_cols:
        return id_map

    buckets = {}
    for _, r in singleton_block.iterrows():
        vals = r[tol_cols].to_numpy(dtype=float, copy=False)
        key = _bucket_key(vals, tol_kinds, tol_vals)
        buckets.setdefault(key, []).append(int(r[id_col]))

    for _, items in buckets.items():
        if len(items) < 2:
            continue
        anchor = min(items)
        for oid in items:
            if oid != anchor:
                id_map[oid] = anchor
    return id_map


def _parse_tol(rule: MatchRule, available_cols: list):
    tol_cols, kinds, vals = [], [], []
    for c, spec in rule.tol_numeric.items():
        if c not in available_cols:
            continue
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] in ("abs", "rel"):
            k, v = spec
        else:
            # retrocompatibilità: numero nudo = assoluto
            k, v = "abs", float(spec)
        tol_cols.append(c)
        kinds.append(k)
        vals.append(float(v))
    return tol_cols, kinds, np.array(vals, dtype=float)



def _process_group(key_vals, group_df: pd.DataFrame, base_ids_set: set, rule: MatchRule, id_col: str, quote_col: str):
    tol_cols, tol_kinds, tol_vals = _parse_tol(rule, group_df.columns.tolist())

    base_block = group_df[group_df["nq"] >= 2].copy()
    singleton_block = group_df[group_df["nq"] == 1].copy()
    if base_block.empty and singleton_block.empty:
        return {}

    id_map = _vectorized_best_match(singleton_block, base_block, tol_cols, tol_kinds, tol_vals, id_col)

    remaining = singleton_block[~singleton_block[id_col].isin(id_map.keys())].copy()
    id_map.update(_cluster_singletons(remaining, tol_cols, tol_kinds, tol_vals, id_col))

    return id_map



def reassign_single_quotation_orders_fast(
    df: pd.DataFrame, id_col: str, quote_col: str, rule: MatchRule,
    min_group_size: int = 2, drop_unmatched: bool = True,
    original_col_name: str = "ordine_originale", verbose: bool = True
) -> pd.DataFrame:
    """Versione ottimizzata della riassegnazione: vettoriale + (opzionale) parallela per gruppi EQUAL_COLS."""
    if id_col not in df.columns or quote_col not in df.columns:
        if verbose: print("[MATCH] id/quotazione non trovate, esco.")
        return df.copy()

    df = df.copy()
    if original_col_name not in df.columns:
        df[original_col_name] = df[id_col]
        if verbose: print(f"[MATCH] Aggiunta colonna '{original_col_name}'.")

    # Precompute counts per id
    cnt = df.groupby(id_col)[quote_col].nunique().rename("nq")
    df = df.merge(cnt, left_on=id_col, right_index=True, how="left")

    # Rappresentanti per le colonne necessarie
    cols_needed = list(set(rule.equal_cols) | set(rule.tol_numeric.keys()) | {id_col})
    reps = _build_representatives(df, id_col, cols_needed).merge(cnt, left_on=id_col, right_index=True, how="left")

    # Chiavi di gruppo (uguali EQUAL_COLS) — velocizza i confronti
    # NB: se una colonna EQUAL_COLS non esiste, la ignoriamo
    eq_cols = [c for c in rule.equal_cols if c in reps.columns]
    if eq_cols:
        reps["_grp_key"] = list(map(tuple, reps[eq_cols].astype(object).to_numpy()))
    else:
        reps["_grp_key"] = ("__all__",) * len(reps)

    # Statistiche di base vs singleton
    base = reps[reps["nq"] >= min_group_size]
    singletons = reps[reps["nq"] == 1]
    if verbose:
        print(f"[MATCH] Gruppi base: {base[id_col].nunique()} | Singleton: {singletons[id_col].nunique()}")

    # Prepara dizionario group_key -> DataFrame slice
    id_map_total = {}
    groups = {k: g.copy() for k, g in reps.groupby("_grp_key", sort=False)}

    if PARALLEL and len(groups) > 1 and MAX_WORKERS > 1:
        if verbose:
            print(f"[PARALLEL] Avvio parallelizzazione su {min(len(groups), MAX_WORKERS)} worker...")

        with ProcessPoolExecutor(max_workers=min(len(groups), MAX_WORKERS)) as ex:
            futures = []
            for key_vals, g in groups.items():
                futures.append(ex.submit(_process_group, key_vals, g, set(), rule, id_col, quote_col))
            for fu in as_completed(futures):
                id_map_total.update(fu.result())
    else:
        for key_vals, g in groups.items():
            id_map_total.update(_process_group(key_vals, g, set(), rule, id_col, quote_col))

    # Applica la mappa ID -> nuovo ID
    if id_map_total:
        df[id_col] = df[id_col].where(~df[id_col].isin(id_map_total.keys()), df[id_col].map(id_map_total))

    # Ricontrolla quante quotazioni post-match
    cnt2 = df.groupby(id_col)[quote_col].nunique().rename("nq_after")
    df = df.drop(columns=["nq"]).merge(cnt2, left_on=id_col, right_index=True, how="left")

    # Drop degli isolati finali (se richiesto)
    if drop_unmatched:
        iso_ids = set(df.loc[df["nq_after"] == 1, id_col].unique().tolist())
        if iso_ids:
            if verbose: print(f"[MATCH] Drop di singleton non simili: {len(iso_ids)}")
            dropped_ids = iso_ids
            df = df[~df[id_col].isin(iso_ids)].copy()
            
    # # Drop degli isolati finali (se richiesto) — SOLO se stato_quotazione == 0
    # if drop_unmatched:
    #     mask_singleton = df["nq_after"] == 1
    #     if "stato_quotazione" in df.columns:
    #         mask_drop = mask_singleton & (df["stato_quotazione"] == 0)
    #         n_drop = int(mask_drop.sum())
    #         if n_drop and verbose:
    #             print(f"[MATCH] Drop di record singleton con stato_quotazione=0: {n_drop}")
    #         df = df[~mask_drop].copy()
    #     else:
    #         # Se la colonna non esiste, mantieni tutto e avvisa
    #         if verbose:
    #             print("[WARN] 'stato_quotazione' non presente: nessun drop dei singleton eseguito.")

    df = df.drop(columns=["nq_after"], errors="ignore")
    return df, dropped_ids


def main():
    input_file = "02_risultati_filtrati.xlsx"
    output_file = "03_risultati_matchati.xlsx"
    input_sheet = "Risultati_filtrati"
    print(f"[LOAD] Leggo da: {input_file}, foglio '{input_sheet}'")
    df_input = pd.read_excel(input_file, sheet_name=input_sheet)
    df_input = df_input.drop(columns="estimated")
    # === Matching ===
    RULE = MatchRule(equal_cols=EQUAL_COLS, tol_numeric=TOL_NUMERIC)
    df_matched, dropped_ids = reassign_single_quotation_orders_fast(
        df_input, ID_COL, QUOTE_COL, RULE, min_group_size=2, drop_unmatched=True, verbose=True
    )

    # === Singleton davvero scartati ===
    df_dropped = df_input[df_input[ID_COL].isin(dropped_ids)]

    # === Salvataggio ===
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_matched.to_excel(writer, sheet_name="Matched", index=False)
            df_dropped.to_excel(writer, sheet_name="Singleton_scartati_match", index=False)
        print(f"[OK] Salvati fogli: Matched, Singleton_scartati_match in {output_file}")
    except Exception as e:
        print(f"[ERRORE] Non sono riuscito a salvare: {e}")



    # Plot (solo matplotlib, più leggero)
    try:
        import matplotlib.pyplot as plt
        os.makedirs("figs", exist_ok=True)

        # n_quotazioni per ordine nel df matched
        nq = (df_matched.groupby(ID_COL)[QUOTE_COL]
                          .nunique(dropna=True)
                          .rename("n_quotazioni"))
        freq = nq.value_counts().sort_index()

        # Grafico assoluto
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()
        ax.bar(freq.index.astype(str), freq.values)
        ax.set_xlabel("Numero di quotazioni per ordine")
        ax.set_ylabel("Numero di ordini")
        ax.set_title("Distribuzione ordini per n_quotazioni — POST MATCH")
        fig.tight_layout()
        fig.savefig(os.path.join("figs", "freq_quotazioni_post_match.png"), dpi=150)
        plt.close(fig)

        # Grafico percentuale
        perc = (freq / freq.sum() * 100).round(2)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.gca()
        ax.bar(perc.index.astype(str), perc.values)
        ax.set_xlabel("Numero di quotazioni per ordine")
        ax.set_ylabel("Percentuale ordini (%)")
        ax.set_title("Distribuzione percentuale ordini per n_quotazioni — POST MATCH")
        for i, v in enumerate(perc.values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join("figs", "freq_quotazioni_post_match_percentuale.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Plot non generati: {e}")

if __name__ == "__main__":
    # Necessario per abilitare multiprocessing su Windows/macOS
    main()
