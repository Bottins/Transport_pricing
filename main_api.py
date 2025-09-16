# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 21:52:43 2025

@author: alexq
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import os
import json
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import traceback

# Importa il nuovo predittore
from prediction_script import RFPredictor

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI(
    title="Transport Price Prediction API v3",
    description="API per predire prezzi di trasporto tramite Random Forest con nuovo preprocessing",
    version="3.0.0"
)

# Crea cartella logs se non esiste
Path("logs").mkdir(exist_ok=True)

# Inizializza il predittore globalmente
predictor = None

@app.on_event("startup")
async def startup_event():
    """Inizializza il predittore all'avvio dell'applicazione"""
    global predictor
    try:
        logger.info("Caricamento modelli in corso...")
        predictor = RFPredictor(models_dir="models")
        logger.info("Modelli caricati con successo!")
    except Exception as e:
        logger.error(f"Errore nel caricamento dei modelli: {e}")
        raise

class TransportRequest(BaseModel):
    """Modello semplificato per richieste di predizione"""
    
    # Dati obbligatori base
    tipo_carico: str
    tipo_trasporto: Union[str, int]  # Può essere numero o stringa
    peso_totale: float
    km_tratta: float
    altezza: float
    lunghezza_max: float
    misure: float
    
    # Località (obbligatori per calcoli)
    naz_carico: str = "IT"
    naz_scarico: str = "IT"
    latitudine_carico: float
    longitudine_carico: float
    latitudine_scarico: float
    longitudine_scarico: float
    
    # Allestimenti (nuova gestione)
    tipi_allestimenti: Optional[Union[str, List[str]]] = None
    specifiche_allestimento: Optional[str] = "base"
    
    # Flags opzionali
    is_isola: str = "no"
    scarico_tassativo: str = "no"
    carico_tassativo: str = "no"
    
    # Date (opzionali, ma utili per il mese)
    data_ordine: Optional[str] = None
    data_carico: Optional[str] = None
    
    @validator('tipo_carico')
    def validate_tipo_carico(cls, v):
        valid_types = ['Completo', 'Parziale', 'Groupage']
        v_cap = v.capitalize()
        if v_cap not in valid_types:
            raise ValueError(f'Tipo carico deve essere uno di: {valid_types}')
        return v_cap
    
    @validator('km_tratta')
    def validate_km_tratta(cls, v):
        if v <= 0 or v > 20000:
            raise ValueError('Km tratta deve essere tra 1 e 20000')
        return v
    
    @validator('peso_totale')
    def validate_peso_totale(cls, v):
        if v <= 0 or v > 100000:
            raise ValueError('Peso totale deve essere tra 1 e 100000 kg')
        return v
    
    @validator('altezza')
    def validate_altezza(cls, v):
        if v <= 0 or v > 500:
            raise ValueError('Altezza deve essere tra 1 e 500 cm')
        return v
    
    @validator('specifiche_allestimento')
    def validate_specifiche_allestimento(cls, v):
        if v is None:
            return "base"
        valid_specs = ['base', 'gru', 'sponda idraulica']
        v_lower = v.lower()
        if v_lower not in valid_specs:
            # Prova a matchare parzialmente
            if 'gru' in v_lower:
                return 'gru'
            elif 'sponda' in v_lower:
                return 'sponda idraulica'
            else:
                return 'base'
        return v_lower
    
    @validator('data_ordine', 'data_carico')
    def validate_dates(cls, v):
        if v is not None:
            try:
                datetime.datetime.strptime(v, '%Y-%m-%d')
                return v
            except ValueError:
                # Se non è nel formato corretto, usiamo data odierna
                return datetime.datetime.now().strftime('%Y-%m-%d')
        # Se è None, usiamo data odierna per avere comunque il mese
        return datetime.datetime.now().strftime('%Y-%m-%d')

class PredictionResponse(BaseModel):
    status: str
    predicted_price: float = None
    confidence_score: float = None
    prediction_std: float = None
    tipo_carico: str = None
    model_version: str = None
    scale_method: str = None
    interval_50_lower: float = None
    interval_50_upper: float = None
    message: str = None
    execution_time: float = None
    
    # Info aggiuntive calcolate
    spazio_calcolato: float = None
    perc_camion: float = None
    tipo_pallet: int = None
    estero: int = None

def log_request(request_data: dict, response_data: dict, execution_time: float):
    """Logga le richieste per debugging"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "request": request_data,
        "response": response_data,
        "execution_time_seconds": execution_time
    }
    
    try:
        with open("logs/prediction_logs_v3.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"Errore nel logging: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: TransportRequest):
    """
    Predice il prezzo di trasporto basato sui dati forniti usando il modello RF v3
    
    Input semplificato: fornisci solo i dati base, le feature derivate vengono calcolate automaticamente.
    """
    start_time = datetime.datetime.now()
    logger.info(f"Iniziando nuova predizione per tipo_carico: {request.tipo_carico}")
    
    try:
        # Converte in dict per il predittore
        prediction_input = request.dict()
        logger.info(f"Dati validati: {prediction_input}")
        
        # Controlla se il predittore è inizializzato
        if predictor is None:
            logger.error("Predittore non inizializzato")
            raise HTTPException(
                status_code=500,
                detail="Predittore non inizializzato"
            )
        
        # Esegui la predizione
        logger.info("Eseguendo predizione con modello RF v3...")
        prediction_result = predictor.predict(prediction_input, return_uncertainty=True)
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Calcola alcune feature per includerle nella risposta (opzionale)
        spazio_calc = None
        perc_camion = None
        tipo_pallet = None
        estero = None
        
        # Calcoli per info aggiuntive
        if request.naz_carico == "IT" and request.naz_scarico == "IT":
            fattore = 0.92
            estero = 0
        else:
            fattore = 0.735
            estero = 1
            
        if request.peso_totale > 0:
            spazio_calc = max(request.peso_totale / fattore, request.misure)
            perc_camion = spazio_calc / 340000 if spazio_calc > 0 else 0
        
        if request.tipo_carico.lower() == 'groupage':
            # Classifica pallet
            h = request.altezza
            p = request.peso_totale
            if h <= 240 and p <= 1200:
                if p <= 350:
                    tipo_pallet = 3  # Ultra Light
                elif p <= 750:
                    tipo_pallet = 2  # Light
                else:
                    tipo_pallet = 1  # Full
            elif h <= 150 and p <= 600:
                if p <= 450:
                    tipo_pallet = 5  # Extra Light
                else:
                    tipo_pallet = 4  # Half
            elif h <= 100 and p <= 300:
                tipo_pallet = 6  # Quarter
            elif h <= 60 and p <= 150:
                tipo_pallet = 7  # Mini Quarter
            else:
                tipo_pallet = 8  # No match
        else:
            tipo_pallet = 0
        
        # Prepara la risposta
        response = PredictionResponse(
            status="success",
            predicted_price=prediction_result["prediction"],
            confidence_score=prediction_result.get("confidence_score"),
            prediction_std=prediction_result.get("prediction_std"),
            tipo_carico=prediction_result["tipo_carico"],
            model_version=prediction_result["model_version"],
            scale_method=prediction_result["scale_method"],
            interval_50_lower=prediction_result.get("interval_50", {}).get("lower"),
            interval_50_upper=prediction_result.get("interval_50", {}).get("upper"),
            message="Predizione completata con successo",
            execution_time=execution_time,
            spazio_calcolato=spazio_calc,
            perc_camion=perc_camion,
            tipo_pallet=tipo_pallet,
            estero=estero
        )
        
        # Log della richiesta
        log_request(prediction_input, response.dict(), execution_time)
        
        logger.info(f"Predizione completata: €{response.predicted_price:.2f} (confidence: {response.confidence_score:.3f})")
        return response
        
    except ValueError as ve:
        logger.error(f"Errore di validazione: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
        
    except Exception as e:
        logger.error(f"Errore imprevisto: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        error_response = {
            "status": "error",
            "message": str(e),
            "execution_time": execution_time
        }
        
        log_request(prediction_input if 'prediction_input' in locals() else {}, error_response, execution_time)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global predictor
    
    models_status = {}
    models_dir = Path("models")
    
    for tipo in ["Completo", "Parziale", "Groupage"]:
        models_status[tipo] = {
            "model_file": models_dir.joinpath(f"rf_model_{tipo}.joblib").exists(),
            "features_file": models_dir.joinpath(f"features_{tipo}.txt").exists(),
            "metadata_file": models_dir.joinpath(f"metadata_{tipo}.json").exists(),
            "gmm_file": models_dir.joinpath(f"gmm_{tipo}.joblib").exists()
        }
    
    predictor_status = predictor is not None
    if predictor_status:
        loaded_models = list(predictor.models.keys())
    else:
        loaded_models = []
    
    return {
        "status": "healthy" if predictor_status else "unhealthy",
        "timestamp": datetime.datetime.now(),
        "service": "Transport Price Prediction API v3",
        "predictor_initialized": predictor_status,
        "loaded_models": loaded_models,
        "models_status": models_status,
        "logs_directory": os.path.exists("logs"),
        "version": "3.0.0"
    }

@app.get("/models/info")
async def models_info():
    """Restituisce informazioni sui modelli caricati"""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predittore non inizializzato")
    
    info = {}
    for tipo in predictor.models.keys():
        info[tipo] = {
            "metadata": predictor.metadata.get(tipo, {}),
            "features_count": len(predictor.features.get(tipo, [])),
            "has_gmm": predictor.gmms.get(tipo) is not None,
            "scale_method": predictor.scale_method.get(tipo, "unknown"),
            "supported_transport_types": predictor.supported_transport_types
        }
    
    return {
        "models_info": info,
        "total_models": len(predictor.models),
        "transport_mapping": predictor.transport_mapping
    }

@app.get("/")
async def root():
    """Endpoint di benvenuto"""
    return {
        "message": "Transport Price Prediction API v3",
        "version": "3.0.0",
        "description": "Modello RF con gestione semplificata allestimenti",
        "supported_types": ["Completo", "Parziale", "Groupage"],
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "models_info": "/models/info (GET)",
            "docs": "/docs (GET)"
        },
        "input_requirements": {
            "mandatory": ["tipo_carico", "tipo_trasporto", "peso_totale", "km_tratta", 
                         "altezza", "lunghezza_max", "misure", "latitudine_carico", 
                         "longitudine_carico", "latitudine_scarico", "longitudine_scarico"],
            "optional": ["tipi_allestimenti", "specifiche_allestimento", "is_isola", 
                        "scarico_tassativo", "carico_tassativo", "data_ordine", "data_carico",
                        "naz_carico", "naz_scarico"],
            "calculated_internally": ["spazio_calcolato", "Perc_camion", "verso_nord", 
                                     "tipo_pallet", "tassativi", "estero", "mese_ordine"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)