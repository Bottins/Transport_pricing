"""
EDA & Preprocessing per dataset ordini Bimob – versione riorganizzata (AGGIORNATA)
----------------------------------------------------------------------------------
Modifiche richieste:
- Tutti i punti in cui si usava l'importo per derivare metriche (es. importo_per_km) ora usano l'importo ATTUALIZZATO
  (colonna "prezzo_attualizzato").
- Aggiunta colonna `importo_per_peso` (prezzo_attualizzato / peso_totale) e relativi grafici/metriche.
- Spostato il calcolo e i filtri/outlier su `importo_per_km` DOPO l'attualizzazione per coerenza.
- NUOVA GESTIONE: tipi_allestimenti e specifiche_allestimento con logiche specifiche e one-hot encoding
- CORREZIONE: Le colonne calcolate appaiono anche negli scartati

Nota: se l'attualizzazione fallisce (mancano file/colonne), si ripiega al comportamento precedente usando `importo`.
"""

# %% Librerie
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# Stile plotting "carino"
sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# %% Utility

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

FIGDIR = "figs"
ensure_dir(FIGDIR)


def save_or_show(fig, name: str | None = None, tight: bool = True):
    if tight:
        fig.tight_layout()
    if name:
        fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.show()


def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Restituisce il primo nome colonna esistente tra i candidati."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- Nuove utility per conteggio quotazioni per ordine ---
ORDER_COL_CANDS = [
    "id_ordine", "idordine", "idOrdine", "ordine_id", "order_id"
]
QUOTE_COL_CANDS = [
    "id_quotazione", "idquotazione", "id_quote", "id_quo",
    "id_preventivo", "idpreventivo", "id_offerta", "offerta_id"
]


def compute_quote_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Calcola n_quotazioni per id ordine. Ritorna (df_conteggi, order_col_name).
    df_conteggi ha colonne [order_col, 'n_quotazioni']."""
    order_col = guess_col(df, ORDER_COL_CANDS)
    quote_col = guess_col(df, QUOTE_COL_CANDS)
    if order_col is None or quote_col is None:
        raise ValueError("Colonne id ordine / id quotazione non trovate. Specifica i nomi o aggiungili ai candidati.")
    counts = (df.groupby(order_col)[quote_col]
                .nunique(dropna=True)
                .rename('n_quotazioni')
                .reset_index())
    return counts, order_col


def ensure_n_quotazioni_column(df: pd.DataFrame) -> pd.DataFrame:
    """Se non esiste una colonna n_quotazioni, la crea a partire da (ordine, quotazione)."""
    if 'n_quotazioni' in df.columns:
        return df
    try:
        counts, order_col = compute_quote_counts(df)
        df = df.merge(counts, on=order_col, how='left')
        return df
    except Exception as e:
        warnings.warn(f"Impossibile derivare n_quotazioni: {e}")
        return df


def plot_quote_count_distribution(df: pd.DataFrame, fase: str, fname_prefix: str):
    """Grafico distribuzione ordini per numero di quotazioni (derivato)."""
    try:
        counts, order_col = compute_quote_counts(df)
        vc = counts['n_quotazioni'].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=vc.index, y=vc.values, ax=ax)
        ax.set_xlabel('Numero quotazioni per ordine')
        ax.set_ylabel('Numero ordini')
        ax.set_title(f'Distribuzione ordini per n_quotazioni — {fase}')
        save_or_show(fig, f"{fname_prefix}_n_quotazioni.png")
    except Exception as e:
        warnings.warn(f"Distribuzione n_quotazioni non prodotta: {e}")
    return None

# %% 1) Caricamento dati

def load_data(path_excel: str = "risultati_ordini.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path_excel)
    return df

# %% 2) EDA iniziale

def eda_before(df: pd.DataFrame):
    print("===== EDA INIZIALE: info / missing / describe =====")
    print(df.info())
    print("Missing per colonna (top 30):", df.isna().sum().sort_values(ascending=False).head(30))

    # Statistiche descrittive per numeriche
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
        print("Statistiche descrittive (numeriche):", desc.head(20))

        # Heatmap di correlazione
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0)
        ax.set_title("Correlazione (numeriche) – PRIMA del preprocessing")
        save_or_show(fig, "corr_before.png")
    else:
        warnings.warn("Poche colonne numeriche per una heatmap significativa.")

    # Distribuzione ordini per numero di quotazioni (derivata da id_ordine/id_quotazione)
    plot_quote_count_distribution(df, fase="PRIMA", fname_prefix="freq_quotazioni_before")

# %% 3) Preprocessing & Feature Engineering

def binary_invert_multilabel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot su colonna lista, poi inverti 0/1 (1=presente)."""
    mlb = MultiLabelBinarizer()
    onehot = mlb.fit_transform(df[col])
    onehot_df = pd.DataFrame(onehot, columns=mlb.classes_, index=df.index)
    return pd.concat([df.drop(columns=[col]), onehot_df], axis=1)


def process_tipi_allestimenti(value):
    """
    Processa il campo tipi_allestimenti secondo la logica:
    - Se "Centinato telonato" è presente nella lista, restituisci solo quello
    - Altrimenti prendi il primo elemento della lista
    """
    if pd.isna(value):
        return "Base"  # Default se NaN
    
    # Converti in stringa e pulisci
    value_str = str(value).strip()
    if not value_str or value_str.lower() == 'nan':
        return "Base"
    
    # Splitta per virgola e pulisci ogni elemento
    elementi = [elem.strip() for elem in value_str.split(',')]
    
    # Controlla se "Centinato telonato" è presente
    if "Centinato telonato" in elementi:
        return "Centinato telonato"
    elif elementi:
        return elementi[0]  # Prendi il primo elemento
    else:
        return "Base"


def process_specifiche_allestimento(value):
    """
    Processa il campo specifiche_allestimento secondo la logica:
    - Se presente "sponda idraulica" o "gru", mantieni quello
    - Altrimenti scrivi "base"
    """
    if pd.isna(value):
        return "base"
    
    # Converti in stringa e pulisci
    value_str = str(value).strip().lower()
    if not value_str or value_str == 'nan':
        return "base"
    
    # Controlla se contiene "sponda idraulica" o "gru"
    if "sponda idraulica" in value_str:
        return "sponda idraulica"
    elif "gru" in value_str:
        return "gru"
    else:
        return "base"


def classifica_pallet(row) -> int:
    if str(row.get('tipo_carico', '')).lower() != 'groupage':
        return 0
    
    try:
        h = int(row.get('altezza', 0))
        p = int(row.get('peso_totale', 0))
        
        a = 8  # Default
        if h <= 240 and p <= 1200:
            if p <= 350:
                a = 3  # Ultra Light Pallet
            elif p <= 750:
                a = 2  # Light Pallet
            else:
                a = 1  # Full Pallet
        if h <= 150 and p <= 600:
            if p <= 450:
                a = 5  # Extra Light Pallet
            else:
                a = 4  # Half Pallet
        if h <= 100 and p <= 300:
            a = 6  # Quarter Pallet
        if h <= 60 and p <= 150:
            a = 7  # Mini Quarter
            
        return a
    except Exception:
        return 8  # Default in caso di errore


def remove_outliers_iqr(group: pd.DataFrame, col: str, k: float = 1.5):
    q1 = group[col].quantile(0.50)
    q3 = group[col].quantile(0.85)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return group[(group[col] >= lower) & (group[col] <= upper)]


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue cleaning + feature + normalizzazione.
    Ritorna (df_raw_clean, df_filtrato_finale).
    """
    # Elimina colonne non necessarie
    colonne_da_eliminare = [
        "idcommittente", "idtrasportatore", 
        "stima_min", "stima_max"
    ]
    df = df.drop(columns=[c for c in colonne_da_eliminare if c in df.columns])

    # =====================
    # NUOVA GESTIONE ALLESTIMENTI
    # =====================
    # Processa tipi_allestimenti
    if 'tipi_allestimenti' in df.columns:
        df['tipi_allestimenti_processed'] = df['tipi_allestimenti'].apply(process_tipi_allestimenti)
        # One-hot encoding per tipi_allestimenti
        tipi_dummies = pd.get_dummies(df['tipi_allestimenti_processed'], prefix='allestimento', dtype='int8')
        df = pd.concat([df, tipi_dummies], axis=1)
        df = df.drop(columns=['tipi_allestimenti', 'tipi_allestimenti_processed'])
    # df = df.drop(columns=['tipi_allestimenti'])
    
    # # Processa specifiche_allestimento
    if 'specifiche_allestimento' in df.columns:
        df['specifiche_allestimento_processed'] = df['specifiche_allestimento'].apply(process_specifiche_allestimento)
        # One-hot encoding per specifiche_allestimento
        spec_dummies = pd.get_dummies(df['specifiche_allestimento_processed'], prefix='specifica', dtype='int8')
        df = pd.concat([df, spec_dummies], axis=1)
        df = df.drop(columns=['specifiche_allestimento', 'specifiche_allestimento_processed'])
    # df = df.drop(columns=['specifiche_allestimento'])

    # Coordinate fuori range
    cols_coord = ["latitudine_carico", "longitudine_carico", "latitudine_scarico", "longitudine_scarico"]
    if all(c in df.columns for c in cols_coord):
        df[cols_coord] = df[cols_coord].mask(df[cols_coord].abs() > 100)
    
    # Colonne con NaN permessi
    colonne_con_nan_permessi = ["estimated", "importotrasp"]

    altre_colonne = [c for c in df.columns if c not in colonne_con_nan_permessi]
    df = df.dropna(subset=[c for c in altre_colonne if c in df.columns])

    # Tipi e filtri base
    df["importotrasp"] = df.get("importotrasp").fillna(df.get("importo"))
    for c in ["importo", "km_tratta", "peso_totale"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "peso_totale" in df.columns:
        df = df[df["peso_totale"] >= 20]
    if "km_tratta" in df.columns:
        df = df[df["km_tratta"] >= 20]
    if "importo" in df.columns:
        df = df[(df["importo"] >= 20) & (df["importo"] <= 7000)]

    # Date
    for c in ["data_ordine", "data_carico"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    if all(c in df.columns for c in ["data_ordine", "data_carico"]):
        df = df[(df['data_ordine'] >= '2016-01-01') & (df['data_carico'] >= '2016-01-01')]
    df = df.drop(columns=["data_carico","data_scarico"])
    
    # Fattore IT/estero e spazio_calcolato
    if all(c in df.columns for c in ["naz_carico", "naz_scarico", "peso_totale", "misure"]):
        fattore = np.where((df["naz_carico"] == "IT") & (df["naz_scarico"] == "IT"), 0.92, 0.735)
        df["spazio_calcolato"] = np.where(
            df["peso_totale"] > 0,
            np.where((df["peso_totale"] / fattore) > df["misure"], df["peso_totale"] / fattore, df["misure"]),
            0,
        )
        
        
    if 'spazio_calcolato' in df.columns:
        df['Perc_camion'] = np.where(
            df['spazio_calcolato'] > 0,
            df['spazio_calcolato']/340000 ,
            np.nan
        )
    
    # Verso
    if all(c in df.columns for c in cols_coord):
        df['verso_nord'] = (df['latitudine_scarico'] - df['latitudine_carico']).astype(float)
        if "verso_nord" in df.columns:
            df = df[(df["verso_nord"] >= -25) & (df["verso_nord"] <= 25)]
        df = df.drop(columns=cols_coord)


    # Pallet & flag vari
    if 'tipo_carico' in df.columns and 'altezza' in df.columns and 'peso_totale' in df.columns:
        df['tipo_pallet'] = df.apply(classifica_pallet, axis=1)

    if 'is_isola' in df.columns:
        df['is_isola'] = df['is_isola'].astype(str).str.lower().eq('si').astype(int)
        if all(c in df.columns for c in ['reg_carico', 'reg_scarico']):
            df.loc[df['reg_carico'] == df['reg_scarico'], 'is_isola'] = 0
    df = df.drop(columns=["misure","reg_carico","reg_scarico"])

    for c in ['scarico_tassativo', 'carico_tassativo']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().eq('si').astype(int)
        # crea la nuova colonna come somma
    df["tassativi"] = df[["scarico_tassativo", "carico_tassativo"]].sum(axis=1)
    df = df.drop(columns=["scarico_tassativo", "carico_tassativo"])


    # Mapping tipo_trasporto -> multilabel binarize
    if 'tipo_trasporto' in df.columns:
        transport_mapping = {
            1: 'Merce generica', 2: 'Temperatura positiva', 3: 'Temperatura negativa',
            4: 'Trasporto auto', 5: 'ADR merce pericolosa', 6: 'Espressi dedicati',
            8: 'Espresso Corriere(plichi-colli)', 9: 'Eccezionali', 10: 'Rifiuti',
            11: 'Via mare', 12: 'Via treno', 13: 'Via aereo', 14: 'Intermodale',
            15: 'Traslochi', 16: 'Cereali sfusi', 17: 'Farmaci', 18: 'Trasporto imbarcazioni',
            19: 'Trasporto pesci vivi', 20: 'Trazioni', 21: 'Noleggio(muletti, ecc.)',
            22: 'Sollevamenti (gru, ecc)', 23: 'Piattaforma-Distribuzione',
            24: 'Operatore doganale', 25: 'Cisternati Chimici', 26: 'Cisternati Carburanti',
            27: 'Cisternati Alimenti', 28: "Opere d'arte"
        }
        df['tipo_trasporto'] = df['tipo_trasporto'].map(transport_mapping)
        df['tipo_trasporto'] = df['tipo_trasporto'].apply(
            lambda x: [s.strip() for s in x.split(',')] if pd.notnull(x) else []
        )
        df = binary_invert_multilabel(df, 'tipo_trasporto')

    # Estero
    if all(c in df.columns for c in ['naz_carico', 'naz_scarico']):
        df['estero'] = df.apply(lambda r: 0 if str(r['naz_carico']).strip() == 'IT' and str(r['naz_scarico']).strip() == 'IT' else 1, axis=1)
        df = df.drop(columns=["naz_carico","naz_scarico"])
    # =====================
    # Attualizzazione
    # =====================
    attualizzazione_ok = False
    try:
        coeff = pd.read_excel('TavoleStream.xlsx', sheet_name='Coefficienti')
        coeff = coeff.iloc[1:].reset_index(drop=True)
        anno_col = coeff.columns[0]
        coeff[anno_col] = pd.to_numeric(coeff[anno_col])
        coeff_long = pd.melt(coeff, id_vars=[anno_col], var_name='Mese', value_name='Coefficiente')
        mesi_map = {'GEN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAG': 5, 'GIU': 6,
                    'LUG': 7, 'AGO': 8, 'SET': 9, 'OTT': 10, 'NOV': 11, 'DIC': 12}
        coeff_long['Mese_Pulito'] = coeff_long['Mese'].astype(str).str.strip().str.upper()
        coeff_long['Mese_Num'] = coeff_long['Mese_Pulito'].map(mesi_map)
        coeff_long = coeff_long.drop(['Mese', 'Mese_Pulito'], axis=1)

        def aggiungi_coefficienti(df_in: pd.DataFrame, colonna_data='data_ordine'):
            di = df_in.copy()
            di[colonna_data] = pd.to_datetime(di[colonna_data])
            di['Anno_Temp'] = di[colonna_data].dt.year
            di['Mese_Temp'] = di[colonna_data].dt.month
            di = di.merge(
                coeff_long,
                left_on=['Anno_Temp', 'Mese_Temp'],
                right_on=[anno_col, 'Mese_Num'],
                how='left'
            )
            di = di.drop(columns=['Anno_Temp', 'Mese_Temp', anno_col, 'Mese_Num'])
            return di

        df = aggiungi_coefficienti(df)
        df = df.dropna(subset=['Coefficiente'])

        df_sorted = df.sort_values('data_ordine')
        prezzo_carb_attuale = df_sorted['prezzo_carb'].iloc[-1]
        coefficiente_attuale = df_sorted['Coefficiente'].iloc[-1]

        df['prezzo_attualizzato'] = (df['importotrasp'] * (0.3*(prezzo_carb_attuale/df['prezzo_carb']) + 0.7*(df['Coefficiente']/coefficiente_attuale))).round(0)
        print(f"Prezzo carburante attuale: {prezzo_carb_attuale}")
        print(f"Coefficiente attuale: {coefficiente_attuale}")
        attualizzazione_ok = True
    except Exception as e:
        warnings.warn(f"Attualizzazione non eseguita: {e}. Si userà l'importo storico per le metriche derivate.")
        df['prezzo_attualizzato'] = df.get('importotrasp', df.get('importo'))

    # =====================
    # IMPORTANTE: Calcolo metriche PRIMA dei filtri
    # =====================
    # Copia dello stato con tutte le feature calcolate ma senza filtri
    df_with_metrics = df.copy()
    
    # importo_per_km basato su 'prezzo_attualizzato'
    if all(c in df_with_metrics.columns for c in ['prezzo_attualizzato', 'km_tratta']):
        df_with_metrics['importo_per_km'] = df_with_metrics['prezzo_attualizzato'] / df_with_metrics['km_tratta']
    
    # importo_per_peso basato su 'prezzo_attualizzato'
    if all(c in df_with_metrics.columns for c in ['prezzo_attualizzato', 'peso_totale']):
        df_with_metrics['importo_per_peso'] = df_with_metrics['prezzo_attualizzato'] / df_with_metrics['peso_totale']
    
    # importo_norm
    if all(c in df_with_metrics.columns for c in ['prezzo_attualizzato', 'km_tratta', 'spazio_calcolato']):
        df_with_metrics['importo_norm'] = np.where(
            df_with_metrics['spazio_calcolato'] >= 1,
            1e5 * df_with_metrics['prezzo_attualizzato'] / (df_with_metrics['km_tratta'] * df_with_metrics['spazio_calcolato']),
            np.nan
        )
    
    # Salva df_clean con tutte le metriche calcolate
    df_clean = df_with_metrics.copy()
    
    # =====================
    # Ora applico i filtri su df per ottenere df_filtered
    # =====================
    df = df_with_metrics.copy()
    
    # Filtri di qualità sulle metriche
    req_cols = ['importo_per_km', 'tipo_carico', 'data_ordine', 'data_carico']
    df = df.dropna(subset=[c for c in req_cols if c in df.columns])
    
    if 'importo_per_km' in df.columns:
        df = df[(df['importo_per_km'] >= 0.15) & (df['importo_per_km'] <= 3.5)]
    
    if 'importo_per_peso' in df.columns and 'tipo_carico' in df.columns:
        print(f"Righe prima del filtro importo_per_peso: {len(df)}")
        df = df[
            (
                ((df['tipo_carico'] == "Groupage") & (df['importo_per_peso'] >= 0.1)) |
                ((df['tipo_carico'] != "Groupage") & (df['importo_per_peso'] >= 0.0))
            ) & 
            (df['importo_per_peso'] <= 10)
        ]
        print(f"Righe dopo il filtro importo_per_peso: {len(df)}")

    if 'importo_norm' in df.columns and 'spazio_calcolato' in df.columns:
        df = df[df['spazio_calcolato'] >= 1]
        df = df.dropna(subset=['importo_norm'])
    
    if 'importo_norm' in df.columns and 'tipo_carico' in df.columns:
        print(f"Righe prima del filtro importo_norm: {len(df)}")
        df = df[
            (
                ((df['tipo_carico'] == "Groupage") & (df['importo_norm'] >= 0.15)) |
                ((df['tipo_carico'] != "Groupage") & (df['importo_norm'] >= 0.0))
            ) & 
            (df['importo_norm'] <= 10)
        ]
        print(f"Righe dopo il filtro importo_norm: {len(df)}")

    # Outlier filtering coerente (dopo attualizzazione)
    if 'tipo_carico' in df.columns and 'importo_per_km' in df.columns:
        df = df.groupby('tipo_carico', group_keys=False).apply(remove_outliers_iqr, col='importo_per_km', k=1)
    
    if 'tipo_carico' in df.columns and 'importo_norm' in df.columns:
        df = df.groupby('tipo_carico', group_keys=False).apply(remove_outliers_iqr, col='importo_norm', k=1)
    
    if 'tipo_carico' in df.columns and 'spazio_calcolato' in df.columns:
        df = df.groupby('tipo_carico', group_keys=False).apply(remove_outliers_iqr, col='spazio_calcolato', k=1)

    # # Keep solo Merce generica == 1 se presente
    # if 'Merce generica' in df.columns:
    #     df = df[df['Merce generica'] == 1]
    #     df = df.loc[:, (df != 0).any(axis=0)]
    #     if 'Merce generica' in df.columns:
    #         df = df.drop(columns=['Merce generica'])

    return df_clean, df

# %% 4) EDA finale e confronti

def _limit_categories(series: pd.Series, top_k: int = 6) -> pd.Series:
    vals = series.value_counts().index.tolist()
    keep = set(vals[:top_k])
    return series.where(series.isin(keep), other='ALTRO')


def dist_overlap(df: pd.DataFrame, col: str, hue: str = 'tipo_carico', title_suffix: str = "", kde: bool = True, bins: int = 50, fname: str | None = None):
    if col not in df.columns or hue not in df.columns:
        warnings.warn(f"Colonna mancante per il grafico: {col} or {hue}")
        return
    tmp = df[[col, hue]].dropna().copy()
    tmp[hue] = _limit_categories(tmp[hue], top_k=6)
    fig, ax = plt.subplots()
    if kde:
        sns.kdeplot(data=tmp, x=col, hue=hue, common_norm=False, fill=True, alpha=0.3, ax=ax)
    else:
        sns.histplot(data=tmp, x=col, hue=hue, element="step", stat="density", common_norm=False, bins=bins, ax=ax)
    ax.set_title(f"Distribuzione {col} {title_suffix}")
    ax.set_xlabel(col)
    ax.set_ylabel("Densità")
    save_or_show(fig, fname)


def eda_after(df_before: pd.DataFrame, df_after: pd.DataFrame):
    print("===== CONFRONTO PRIMA vs DOPO =====")
    print("Righe prima:", len(df_before), " — Righe dopo:", len(df_after))

    df_before = ensure_n_quotazioni_column(df_before)
    df_after  = ensure_n_quotazioni_column(df_after)

    target_cols = [c for c in ["importo_per_km", "importo_per_peso", "importo_norm", "spazio_calcolato"] if c in df_after.columns or c in df_before.columns]

    def by_tipo_summary(dfi: pd.DataFrame, label: str):
        cols = [c for c in target_cols if c in dfi.columns]
        if not cols or 'tipo_carico' not in dfi.columns:
            return None
        agg = dfi.groupby('tipo_carico')[cols].agg(['count', 'median', 'mean', 'std']).round(3)
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg.insert(0, 'fase', label)
        return agg

    a = by_tipo_summary(df_before, 'PRIMA')
    b = by_tipo_summary(df_after, 'DOPO')
    if a is not None and b is not None:
        comp = pd.concat([a, b])
        print("Statistiche per tipo_carico (prima/dopo):", comp.head(20))

    # Heatmap correlazione finale
    num_cols = df_after.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df_after[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0)
        ax.set_title("Correlazione – DOPO il preprocessing")
        save_or_show(fig, "corr_after.png")

    # Distribuzione ordini per n_quotazioni (derivata) — finale
    plot_quote_count_distribution(df_after, fase="DOPO", fname_prefix="freq_quotazioni_after")

    # Distribuzioni sovrapposte richieste
    if 'tipo_carico' in df_after.columns:
        if 'importo_per_km' in df_after.columns:
            dist_overlap(df_after, 'importo_per_km', 'tipo_carico', title_suffix='— DOPO', kde=True, fname='dist_importo_per_km_after.png')
        if 'importo_per_peso' in df_after.columns:
            dist_overlap(df_after, 'importo_per_peso', 'tipo_carico', title_suffix='— DOPO', kde=True, fname='dist_importo_per_peso_after.png')
        if 'importo_norm' in df_after.columns:
            dist_overlap(df_after, 'importo_norm', 'tipo_carico', title_suffix='— DOPO', kde=True, fname='dist_importo_norm_after.png')
        if 'spazio_calcolato' in df_after.columns:
            dist_overlap(df_after, 'spazio_calcolato', 'tipo_carico', title_suffix='— DOPO', kde=True, fname='dist_spazio_calcolato_after.png')

    # Report colonne allestimenti create
    print("\n===== COLONNE ALLESTIMENTI CREATE =====")
    allestimento_cols = [c for c in df_after.columns if c.startswith('allestimento_') or c.startswith('specifica_')]
    if allestimento_cols:
        print(f"Trovate {len(allestimento_cols)} colonne allestimenti:")
        for col in allestimento_cols:
            print(f"  - {col}: {df_after[col].sum()} records con valore 1")

# %% 5) Pipeline principale

def main():
    # Carica
    input_file = "01_risultati_ordini.xlsx"
    output_file = "02_risultati_filtrati.xlsx"
    df = load_data(input_file)

    # EDA iniziale
    eda_before(df)

    # Preprocess
    df_before, df_after = preprocess(df)
    
    # Calcolo scartati con merge
    scartati = df_before.merge(df_after, how="outer", indicator=True)
    scartati = scartati[scartati["_merge"] == "left_only"].drop(columns=["_merge"])
    
    print(f"\n===== RIGHE SCARTATE: {len(scartati)} =====")
    
    # Verifica che le colonne calcolate siano presenti negli scartati
    cols_to_check = ['importo_per_km', 'importo_per_peso', 'importo_norm', 'Perc_camion']
    cols_present = [c for c in cols_to_check if c in scartati.columns]
    print(f"Colonne calcolate presenti negli scartati: {cols_present}")
    
    # EDA finale (confronto)
    eda_after(df_before, df_after)

    # Salvataggio export finale nello stesso file di input, ma in un nuovo foglio
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_after.to_excel(writer, sheet_name="Risultati_filtrati", index=False)
            scartati.to_excel(writer, sheet_name="Scartati", index=False)
        print(f"\nDati salvati in '{output_file}' nei fogli 'Risultati_filtrati' e 'Scartati'")
        print(f"Le colonne calcolate sono ora presenti in entrambi i fogli.")
    except Exception as e:
        warnings.warn(f"Export non riuscito: {e}")


if __name__ == "__main__":
    main()