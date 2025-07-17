# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 14:08:49 2025

@author: alexq
"""

#!/usr/bin/env python3
"""
Script di predizione prezzi trasporto usando Dual Random Forest
Riceve dati JSON dal form web e restituisce predizioni
"""

import sys
import json
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

class DualRFPredictor:
    def __init__(self, model_path='dual_rf_models_TRASP_best.pkl'):
        """
        Inizializza il predittore caricando i modelli salvati
        """
        self.model_path = model_path
        self.models_dict = None
        self.threshold = None
        self.load_models()
    
    def load_models(self):
        """Carica i modelli salvati dal file pickle"""
        try:
            with open(self.model_path, 'rb') as f:
                self.models_dict = pickle.load(f)
            
            self.threshold = self.models_dict['threshold']
            print(f" Modelli caricati con successo!")
            print(f"   - Soglia di divisione: {self.threshold}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File modello non trovato: {self.model_path}")
        except Exception as e:
            raise Exception(f"Errore nel caricamento del modello: {str(e)}")
    
    def binary_invert_multilabel(self, df, col, expected_classes=None):
        """One-hot encoding su colonna con liste"""
        if col not in df.columns:
            print(f"  Colonna '{col}' non trovata, saltando...")
            return df
        
        mlb = MultiLabelBinarizer()
        onehot = mlb.fit_transform(df[col])
        onehot_df = pd.DataFrame(onehot, columns=mlb.classes_, index=df.index)
        
        if expected_classes is not None:
            for cls in expected_classes:
                if cls not in onehot_df.columns:
                    onehot_df[cls] = 0
            onehot_df = onehot_df.reindex(columns=expected_classes, fill_value=0)
        
        return pd.concat([df.drop(columns=[col]), onehot_df], axis=1)
    
    def preprocess_prediction_data(self, df):
        """Preprocessa i dati per la predizione"""
        print(" Inizio preprocessing dei dati...")
        
        df = df.copy()
        
        # 1. Processa date e crea mese/anno carico
        if 'data_carico' in df.columns:
            df['data_carico'] = pd.to_datetime(df['data_carico'], errors='coerce')
            df = df.dropna(subset=['data_carico'])
            df['mesecarico'] = df['data_carico'].dt.month
            df['annocarico'] = df['data_carico'].dt.year
        else:
            df['mesecarico'] = 6
            df['annocarico'] = 2024
        
        # 2. Processa tipi_allestimenti
        if 'tipi_allestimenti' in df.columns:
            df['tipi_allestimenti'] = df['tipi_allestimenti'].apply(
                lambda x: [s.strip() for s in x.split(',')][:-1] if pd.notnull(x) and x.strip() else []
            )
            df = self.binary_invert_multilabel(df, 'tipi_allestimenti')
        
        # 3. Crea flag estero
        if 'naz_carico' in df.columns and 'naz_scarico' in df.columns:
            df['estero'] = df.apply(
                lambda r: 0.092 if str(r['naz_carico']).strip() == 'IT' and
                                 str(r['naz_scarico']).strip() == 'IT' else 0.0735, axis=1
            )
        else:
            df['estero'] = 0.092
        
        # 4. Calcola verso_nord
        if 'latitudine_scarico' in df.columns and 'latitudine_carico' in df.columns:
            df['verso_nord'] = ((df['latitudine_scarico'] - df['latitudine_carico'])).astype(float)
            df['verso_nord'] = (df['verso_nord'] >= 1).astype(int)
        else:
            df['verso_nord'] = 0.0
        
        # 5. Processa tipo_carico e calcola tipo_pallet
        def classifica_pallet(row):
            if row['tipo_carico'].lower() != 'groupage':
                return 0
            
            h = row['altezza']
            p = row['peso_totale']
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
        
        df['tipo_pallet'] = df.apply(classifica_pallet, axis=1)
        
        if 'tipo_carico' in df.columns:
            df['tipo_carico'] = df['tipo_carico'].apply(
                lambda x: [s.strip() for s in x.split(',')] if pd.notnull(x) else []
            )
            df = self.binary_invert_multilabel(df, 'tipo_carico')
        
        df["Groupage"] = df['tipo_pallet']
        
        # 6. Processa specifiche_allestimento
        if 'specifiche_allestimento' in df.columns:
            df['specifiche_allestimento'] = df['specifiche_allestimento'].apply(
                lambda x: [s.strip() for s in x.split(',')] if pd.notnull(x) and x.strip() else []
            )
            df = self.binary_invert_multilabel(df, 'specifiche_allestimento')
        
        # 7. Processa flag booleani
        if 'is_isola' in df.columns:
            df['is_isola'] = df['is_isola'].str.lower().eq('si').astype(int)
            if 'reg_carico' in df.columns and 'reg_scarico' in df.columns:
                df.loc[df['reg_carico'] == df['reg_scarico'], 'is_isola'] = 0
        else:
            df['is_isola'] = 0
        
        if 'scarico_tassativo' in df.columns:
            df['scarico_tassativo'] = df['scarico_tassativo'].str.lower().eq('si').astype(int)
        else:
            df['scarico_tassativo'] = 0
            
        if 'carico_tassativo' in df.columns:
            df['carico_tassativo'] = df['carico_tassativo'].str.lower().eq('si').astype(int)
        else:
            df['carico_tassativo'] = 0
        
        # 8. Processa tipo_trasporto
        transport_mapping = {
            1: 'Merce generica', 2: 'Temperatura positiva', 3: 'Temperatura negativa',
            4: 'Trasporto auto', 5: 'ADR merce pericolosa', 6: 'Espressi dedicati',
            8: 'Espresso Corriere(plichi-colli)', 9: 'Eccezionali', 10: 'Rifiuti',
            11: 'Via mare', 12: 'Via treno', 13: 'Via aereo', 14: 'Intermodale',
            15: 'Traslochi', 16: 'Cereali sfusi', 17: 'Farmaci', 18: 'Trasporto imbarcazioni',
            19: 'Trasporto pesci vivi', 20: 'Trazioni', 21: 'Noleggio(muletti, ecc.)',
            22: 'Sollevamenti (gru, ecc)', 23: 'Piattaforma-Distribuzione',
            24: 'Operatore doganale', 25: 'Cisternati Chimici', 26: 'Cisternati Carburanti',
            27: 'Cisternati Alimenti', 28: 'Opere d\'arte'
        }
        
        if 'tipo_trasporto' in df.columns:
            # Se è un numero, mappalo
            if df['tipo_trasporto'].dtype in ['int64', 'float64']:
                df['tipo_trasporto'] = df['tipo_trasporto'].map(transport_mapping)
            
            df['tipo_trasporto'] = df['tipo_trasporto'].apply(
                lambda x: [s.strip() for s in x.split(',')] if pd.notnull(x) else []
            )
            df = self.binary_invert_multilabel(df, 'tipo_trasporto')
        
        # 9. Rimuovi colonne non più utili
        drop_cols = ['data_carico', 'tipo_pallet', 'naz_carico', 'naz_scarico', 'reg_carico', 'reg_scarico']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # 10. Applica clipping e trasformazioni
        if 'km_tratta' in df.columns:
            df['km_tratta'] = df['km_tratta'].clip(lower=30, upper=15000)
        
        if 'peso_totale' in df.columns:
            df['peso_totale'] = df['peso_totale'].clip(lower=30, upper=80000)
        
        if 'prezzo_carb' in df.columns:
            df['prezzo_carb'] = df['prezzo_carb'] / 320
        
        if 'misure' in df.columns:
            df['misure'] = df['misure'] / 100
        
        # 11. Crea feature aggiuntive
        if 'km_tratta' in df.columns and 'peso_totale' in df.columns:
            df['km_peso_product'] = df['km_tratta'] * df['peso_totale']
            df['peso_km_rapport'] = df['peso_totale'] / df['km_tratta']
            df['peso_log'] = np.log(df['peso_totale'])
        
        print(f" Preprocessing completato! Shape finale: {df.shape}")
        return df
    
    def align_features(self, df, expected_features):
        """Allinea le features del DataFrame con quelle attese dal modello"""
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        df_aligned = df[expected_features].copy()
        return df_aligned
    
    def predict_single(self, input_data):
        """Predice per un singolo record"""
        if self.models_dict is None:
            raise Exception("Modelli non caricati!")
        
        # Crea DataFrame da input singolo
        df = pd.DataFrame([input_data])
        
        # Preprocessa
        df_processed = self.preprocess_prediction_data(df)
        
        # Determina quale modello usare
        split_condition = (
            (df_processed['peso_totale'] < 20*self.threshold) + 
            (df_processed['km_tratta'] < self.threshold) 
        )
        
        use_small_model = split_condition.iloc[0]
        
        if use_small_model:
            # Usa Small RF
            df_aligned = self.align_features(df_processed, self.models_dict['features_small'])
            X = df_aligned.values.astype(float)
            X_scaled = self.models_dict['scaler_small'].transform(X)
            
            pred = self.models_dict['rf_small'].predict(X_scaled)[0]
            pred = round(pred, -1)  # Arrotonda alle decine
            
            # Calcola confidence
            individual_preds = np.array([
                tree.predict(X_scaled)[0] 
                for tree in self.models_dict['rf_small'].estimators_
            ])
            confidence_std = np.std(individual_preds)
            max_std = 200  # Valore approssimativo
            confidence = max(0, 1 - (confidence_std / max_std))
            
            model_used = 'Small_RF'
        else:
            # Usa Large RF
            df_aligned = self.align_features(df_processed, self.models_dict['features_large'])
            X = df_aligned.values.astype(float)
            X_scaled = self.models_dict['scaler_large'].transform(X)
            
            pred = self.models_dict['rf_large'].predict(X_scaled)[0]
            pred = round(pred, -1)  # Arrotonda alle decine
            
            # Calcola confidence
            individual_preds = np.array([
                tree.predict(X_scaled)[0] 
                for tree in self.models_dict['rf_large'].estimators_
            ])
            confidence_std = np.std(individual_preds)
            max_std = 500  # Valore approssimativo
            confidence = max(0, 1 - (confidence_std / max_std))
            
            model_used = 'Large_RF'
        
        # Calcola range di predizione
        confidence_factor = 1 - confidence
        base_uncertainty = 0.1
        max_uncertainty = 0.60
        uncertainty_range = base_uncertainty + (max_uncertainty - base_uncertainty) * confidence_factor
        
        pred_min = round(pred * (1 - uncertainty_range/2), -1)
        pred_max = round(pred * (1 + uncertainty_range), -1)
        
        return {
            'predicted_price': float(pred),
            'confidence_score': float(confidence),
            'model_used': model_used,
            'price_range_min': float(pred_min),
            'price_range_max': float(pred_max),
            'uncertainty_percentage': float(uncertainty_range * 100)
        }

def process_transport_data(data):
    """
    Processa i dati di trasporto e restituisce la predizione
    """
    print(f"  Processando richiesta di predizione...")
    print(f"   - Tratta: {data.get('km_tratta', 'N/A')} km")
    print(f"   - Peso: {data.get('peso_totale', 'N/A')} kg")
    print(f"   - Tipo trasporto: {data.get('tipo_trasporto', 'N/A')}")
    
    try:
        # Inizializza il predittore
        predictor = DualRFPredictor()
        
        # Esegui la predizione
        prediction_result = predictor.predict_single(data)
        
        print(f" Predizione completata!")
        print(f"   - Prezzo stimato: €{prediction_result['predicted_price']:.2f}")
        print(f"   - Confidence: {prediction_result['confidence_score']:.3f}")
        print(f"   - Modello usato: {prediction_result['model_used']}")
        print(f"   - Range: €{prediction_result['price_range_min']:.2f} - €{prediction_result['price_range_max']:.2f}")
        
        return prediction_result
        
    except Exception as e:
        print(f" Errore nella predizione: {str(e)}")
        raise

def main():
    """
    Funzione principale che riceve i dati dal form
    """
    try:
        # Ricevi i dati JSON dal command line
        if len(sys.argv) != 2:
            print("Errore: Script richiede esattamente un argomento JSON")
            sys.exit(1)
        
        json_data = sys.argv[1]
        data = json.loads(json_data)
        
        # Processa i dati e ottieni la predizione
        prediction_result = process_transport_data(data)
        
        # Stampa il risultato in formato che l'API può parsare
        print("PREDICTION_RESULT:" + json.dumps(prediction_result, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        print(f"Errore nel parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Errore nell'elaborazione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()