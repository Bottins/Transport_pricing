from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import os
import json
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Importa il nuovo predittore
from prediction_script import RFPredictor

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI(
    title="Transport Price Prediction API v2",
    description="API per predire prezzi di trasporto tramite Random Forest (Nuovo Modello)",
    version="2.0.0"
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
    # Dati obbligatori
    tipo_carico: str
    tipo_trasporto: str
    peso_totale: float
    km_tratta: float
    altezza: float
    lunghezza_max: float
    misure: float
    naz_carico: str = "IT"
    naz_scarico: str = "IT"
    reg_carico: str
    reg_scarico: str
    latitudine_carico: float
    longitudine_carico: float
    latitudine_scarico: float
    longitudine_scarico: float
    
    # Dati opzionali con valori di default
    is_isola: str = "no"
    scarico_tassativo: str = "no"
    carico_tassativo: str = "no"
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
    
    @validator('data_ordine', 'data_carico')
    def validate_dates(cls, v):
        if v is not None:
            try:
                datetime.datetime.strptime(v, '%Y-%m-%d')
                return v
            except ValueError:
                raise ValueError('Le date devono essere nel formato YYYY-MM-DD')
        return v

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
        with open("logs/prediction_logs_v2.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"Errore nel logging: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: TransportRequest):
    """
    Predice il prezzo di trasporto basato sui dati forniti usando il nuovo modello RF
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
        logger.info("Eseguendo predizione con nuovo modello RF...")
        prediction_result = predictor.predict(prediction_input, return_uncertainty=True)
        
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
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
            execution_time=execution_time
        )
        
        # Log della richiesta
        log_request(prediction_input, response.dict(), execution_time)
        
        logger.info(f"Predizione completata: €{response.predicted_price:.2f} (confidence: {response.confidence_score:.3f})")
        return response
        
    except ValueError as ve:
        # Errori di validazione dei dati
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
    
    # Controlla esistenza file modelli
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
        "service": "Transport Price Prediction API v2",
        "predictor_initialized": predictor_status,
        "loaded_models": loaded_models,
        "models_status": models_status,
        "logs_directory": os.path.exists("logs")
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
            "scale_method": predictor.scale_method.get(tipo, "unknown")
        }
    
    return {
        "models_info": info,
        "total_models": len(predictor.models)
    }

@app.get("/")
async def root():
    """Endpoint di benvenuto"""
    return {
        "message": "Transport Price Prediction API v2",
        "version": "2.0.0",
        "description": "Nuovo modello RF con predizioni per tipo carico",
        "supported_types": ["Completo", "Parziale", "Groupage"],
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "models_info": "/models/info (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)