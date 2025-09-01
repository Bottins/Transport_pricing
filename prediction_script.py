#!/usr/bin/env python3
"""
Modulo predittore per modelli Random Forest.
Estratto dal codice 05_predict.py per essere utilizzato come modulo nell'API.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict

class RFPredictor:
    """Predittore per modelli RF con preprocessing integrato + GMM per confidence."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.features = {}
        self.metadata = {}
        self.gmms = {}          # GMM per tipo
        self.scale_method = {}  # metodo di scala per tipo (dai metadata)

        for tipo in ["Completo", "Parziale", "Groupage"]:
            self._load_model(tipo)

    def _load_model(self, tipo: str):
        """Carica modello, features, metadata e GMM per un tipo specifico"""
        model_path = os.path.join(self.models_dir, f"rf_model_{tipo}.joblib")
        features_path = os.path.join(self.models_dir, f"features_{tipo}.txt")
        meta_path = os.path.join(self.models_dir, f"metadata_{tipo}.json")
        gmm_path = os.path.join(self.models_dir, f"gmm_{tipo}.joblib")

        if os.path.exists(model_path):
            self.models[tipo] = joblib.load(model_path)
            
            with open(features_path, 'r', encoding="utf-8") as f:
                self.features[tipo] = [line.strip() for line in f.readlines()]
            
            with open(meta_path, 'r', encoding="utf-8") as f:
                self.metadata[tipo] = json.load(f)

            # Carica GMM se esiste
            if os.path.exists(gmm_path):
                try:
                    self.gmms[tipo] = joblib.load(gmm_path)
                except Exception:
                    self.gmms[tipo] = None
            else:
                self.gmms[tipo] = None

            # Memorizza scale method
            self.scale_method[tipo] = self.metadata[tipo].get("scale_method", "gmm_global")
            
            print(f"Modello {tipo} caricato con successo")
        else:
            print(f"Modello {tipo} non trovato in {model_path}")

    def preprocess_row(self, data: dict) -> pd.DataFrame:
        """Preprocessing dei dati di input"""
        df = pd.DataFrame([data])
        
        # Conversioni numeriche con maggiore compatibilità NumPy 2.x
        numeric_cols = ["importo", "km_tratta", "peso_totale", "altezza", 
                        "lunghezza_max", "misure"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype('float64')
        
        # Conversioni date
        for col in ["data_ordine", "data_carico"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if "data_ordine" in df.columns:
            df["mese_ordine"] = pd.to_datetime(df["data_ordine"]).dt.month

        # Calcolo spazio_calcolato
        if all(c in df.columns for c in ["naz_carico", "naz_scarico", "peso_totale", "misure"]):
            fattore = np.where(
                (df["naz_carico"] == "IT") & (df["naz_scarico"] == "IT"), 
                0.92, 0.735
            )
            df["spazio_calcolato"] = np.where(
                df["peso_totale"] > 0,
                np.where((df["peso_totale"] / fattore) > df["misure"], 
                         df["peso_totale"] / fattore, df["misure"]),
                0
            )

        # Calcolo direzioni geografiche
        if all(c in df.columns for c in ["latitudine_scarico", "latitudine_carico", 
                                         "longitudine_scarico", "longitudine_carico"]):
            df['verso_nord'] = (df['latitudine_scarico'] - df['latitudine_carico']).astype(float)
            df['verso_est'] = (df['longitudine_scarico'] - df['longitudine_carico']).astype(float)

        # Classificazione pallet per Groupage
        if all(c in df.columns for c in ['tipo_carico', 'altezza', 'peso_totale']):
            df['tipo_pallet'] = df.apply(self._classifica_pallet, axis=1)

        # Conversione flag booleani
        for col in ['is_isola', 'scarico_tassativo', 'carico_tassativo']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().eq('si').astype(int)

        # Flag estero
        if all(c in df.columns for c in ['naz_carico', 'naz_scarico']):
            df['estero'] = df.apply(
                lambda r: 0 if str(r['naz_carico']).strip() == 'IT' and 
                              str(r['naz_scarico']).strip() == 'IT' else 1, 
                axis=1
            )

        # Mappatura tipi trasporto
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
        
        all_transport_types = [
            'ADR merce pericolosa', 'Eccezionali', 'Espressi dedicati',
            'Espresso Corriere(plichi-colli)', 'Intermodale', 'Merce generica',
            'Rifiuti', 'Temperatura negativa', 'Temperatura positiva',
            'Traslochi', 'Trasporto auto', 'Trasporto imbarcazioni',
            'Via aereo', 'Via mare'
        ]

        # Inizializza tutte le colonne di trasporto a 0
        for transport_type in all_transport_types:
            df[transport_type] = 0

        # Processa il tipo_trasporto
        if 'tipo_trasporto' in df.columns:
            tipo_value = df.iloc[0]['tipo_trasporto']
            if pd.notnull(tipo_value):
                if isinstance(tipo_value, (int, float)):
                    tipo_str = transport_mapping.get(int(tipo_value), '')
                    if tipo_str in all_transport_types:
                        df[tipo_str] = 1
                elif isinstance(tipo_value, str):
                    # Prova prima a convertire in numero
                    try:
                        tipo_num = int(tipo_value)
                        tipo_str = transport_mapping.get(tipo_num, '')
                        if tipo_str in all_transport_types:
                            df[tipo_str] = 1
                    except ValueError:
                        # Se non è un numero, processa come stringa
                        types = [s.strip() for s in tipo_value.split(',')]
                        for t in types:
                            if t in all_transport_types:
                                df[t] = 1
        
        return df

    def _classifica_pallet(self, row) -> int:
        """Classifica il tipo di pallet per il Groupage"""
        if str(row.get('tipo_carico', '')).lower() != 'groupage':
            return 0
        
        h = row.get('altezza', 0)
        p = row.get('peso_totale', 0)
        
        try:
            if h <= 240 and p <= 1200:
                if p <= 350:
                    return 3  # Ultra Light Pallet
                elif p <= 750:
                    return 2  # Light Pallet
                else:
                    return 1  # Full Pallet
            elif h <= 150 and p <= 600:
                if p <= 450:
                    return 5  # Extra Light Pallet
                else:
                    return 4  # Half Pallet
            elif h <= 100 and p <= 300:
                return 6  # Quarter Pallet
            elif h <= 60 and p <= 150:
                return 7  # Mini Quarter
            else:
                return 8  # Nessuna corrispondenza
        except:
            return 8

    def _gmm_mixture_std(self, gm) -> float:
        """Calcola la deviazione standard della miscela GMM"""
        if gm is None:
            return None
        
        w = gm.weights_
        mu = gm.means_.flatten()
        # Compatibilità NumPy 2.x - gestione più robusta dei tipi
        sigma2 = np.array([np.squeeze(gm.covariances_[i]) for i in range(gm.n_components)], dtype=np.float64)
        
        mu_mix = np.sum(w * mu)
        var_mix = np.sum(w * (sigma2 + (mu - mu_mix) ** 2))
        
        return float(np.sqrt(np.maximum(var_mix, 1e-12)))

    def _gmm_local_std(self, gm, y_hat: float) -> float:
        """Calcola la deviazione standard locale del GMM"""
        if gm is None:
            return None
        
        r = gm.predict_proba(np.array([[float(y_hat)]], dtype=np.float64))[0]
        k = int(np.argmax(r))
        sigma2_k = float(np.squeeze(gm.covariances_[k]))
        
        return float(np.sqrt(np.maximum(sigma2_k, 1e-12)))

    def predict(self, data: dict, return_uncertainty: bool = True) -> Dict:
        """
        Esegue la predizione per un singolo record
        
        Args:
            data: Dizionario con i dati di input
            return_uncertainty: Se calcolare metriche di incertezza
        
        Returns:
            Dizionario con i risultati della predizione
        """
        tipo_carico = data.get("tipo_carico", "").capitalize()
        
        if tipo_carico not in self.models:
            raise ValueError(f"Tipo carico '{tipo_carico}' non supportato. Usa: {list(self.models.keys())}")

        # Preprocessing
        df = self.preprocess_row(data)
        
        # Estrai features per il modello
        feature_cols = self.features[tipo_carico]
        
        # Assicurati che tutte le features esistano
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols]
        
        # Predizione
        pipe = self.models[tipo_carico]
        y_pred = float(pipe.predict(X)[0])

        result = {
            "prediction": y_pred,
            "tipo_carico": tipo_carico,
            "model_version": self.metadata[tipo_carico].get("version", "1.0"),
            "scale_method": self.scale_method.get(tipo_carico, "gmm_global")
        }

        if return_uncertainty:
            # Calcola metriche di incertezza
            rf = pipe.named_steps["rf"]
            preproc = pipe.named_steps["prep"]
            
            X_transformed = preproc.transform(X)
            # Compatibilità NumPy 2.x - uso dtype esplicito
            tree_preds = np.array([tree.predict(X_transformed) for tree in rf.estimators_], dtype=np.float64)
            
            std_pred = float(np.std(tree_preds))
            median_pred = float(np.median(tree_preds))

            gm = self.gmms.get(tipo_carico, None)
            scale_method = self.scale_method.get(tipo_carico, "gmm_global")

            if gm is not None:
                if scale_method == "gmm_local":
                    scale = self._gmm_local_std(gm, y_pred)
                else:
                    scale = self._gmm_mixture_std(gm)
            else:
                # Fallback se GMM non disponibile
                scale = max(median_pred * 0.1, 100.0)

            # Calcola confidence score - uso np.maximum per NumPy 2.x
            denom = std_pred + float(scale) + 1e-9
            confidence = 1.0 - (std_pred / denom)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            # Intervallo di confidenza
            lower = float(np.percentile(tree_preds, 25))
            upper = float(np.percentile(tree_preds, 75))

            result.update({
                "confidence_score": confidence,
                "prediction_std": std_pred,
                "interval_50": {"lower": lower, "upper": upper}
            })

        return result