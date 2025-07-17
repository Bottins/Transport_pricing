from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import os
import json
import datetime
import subprocess
import logging
from pathlib import Path

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI(
    title="Transport Price Prediction API",
    description="API per predire prezzi di trasporto tramite Random Forest",
    version="1.0.0"
)

# Crea cartella logs se non esiste
Path("logs").mkdir(exist_ok=True)

class TransportRequest(BaseModel):
    # Dati obbligatori
    data_carico: str
    latitudine_carico: float
    longitudine_carico: float
    latitudine_scarico: float
    longitudine_scarico: float
    naz_carico: str = "IT"
    naz_scarico: str = "IT"
    reg_carico: str
    reg_scarico: str
    tipo_trasporto: str
    tipo_carico: str
    km_tratta: float
    peso_totale: float
    misure: float
    altezza: float
    lunghezza_max: float
    prezzo_carb: float
    
    # Dati opzionali
    carico_tassativo: str = "no"
    scarico_tassativo: str = "no"
    tipi_allestimenti: str = ""
    specifiche_allestimento: str = ""
    is_isola: str = "no"
    note: str = ""
    
    @validator('data_carico')
    def validate_data_carico(cls, v):
        try:
            datetime.datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Data carico deve essere nel formato YYYY-MM-DD')
    
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

class PredictionResponse(BaseModel):
    status: str
    predicted_price: float = None
    confidence_score: float = None
    model_used: str = None
    price_range_min: float = None
    price_range_max: float = None
    uncertainty_percentage: float = None
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
        with open("logs/prediction_logs.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Errore nel logging: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: TransportRequest):
    """
    Predice il prezzo di trasporto basato sui dati forniti
    """
    start_time = datetime.datetime.now()
    logger.info("Iniziando nuova predizione")
    
    try:
        # Converte in dict per lo script
        prediction_input = request.dict()
        logger.info(f"Dati validati: {prediction_input}")
        
        # Controlla se i file necessari esistono
        if not os.path.exists("prediction_script.py"):
            logger.error("File prediction_script.py non trovato")
            raise HTTPException(
                status_code=500,
                detail="File prediction_script.py non trovato"
            )
        
        if not os.path.exists("models/dual_rf_models_TRASP_best.pkl"):
            logger.error("File modello non trovato")
            raise HTTPException(
                status_code=500,
                detail="File modello non trovato"
            )
        
        # Esegui lo script di predizione
        logger.info("Eseguendo script di predizione...")
        cmd = ["python", "prediction_script.py", json.dumps(prediction_input)]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd()
        )
        
        logger.info(f"Return code: {result.returncode}")
        logger.info(f"STDOUT: {result.stdout}")
        
        if result.returncode == 0:
            # Parsing dell'output dello script
            output_lines = result.stdout.strip().split('\n')
            prediction_result = None
            
            for line in output_lines:
                if line.startswith('PREDICTION_RESULT:'):
                    try:
                        prediction_result = json.loads(line.replace('PREDICTION_RESULT:', ''))
                        logger.info(f"Prediction result parsed: {prediction_result}")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"Errore nel parsing del JSON: {e}")
            
            if prediction_result:
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                
                response = PredictionResponse(
                    status="success",
                    predicted_price=prediction_result.get("predicted_price"),
                    confidence_score=prediction_result.get("confidence_score"),
                    model_used=prediction_result.get("model_used"),
                    price_range_min=prediction_result.get("price_range_min"),
                    price_range_max=prediction_result.get("price_range_max"),
                    uncertainty_percentage=prediction_result.get("uncertainty_percentage"),
                    message="Predizione completata con successo",
                    execution_time=execution_time
                )
                
                # Log della richiesta
                log_request(prediction_input, response.dict(), execution_time)
                
                return response
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Errore nel parsing del risultato della predizione"
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Script di predizione terminato con errore: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        logger.error("Timeout nella predizione")
        raise HTTPException(
            status_code=408,
            detail="Timeout nella predizione (>60s)"
        )
    except Exception as e:
        logger.error(f"Errore imprevisto: {str(e)}")
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
    files_status = {
        "prediction_script.py": os.path.exists("prediction_script.py"),
        "models/dual_rf_models_TRASP_best.pkl": os.path.exists("models/dual_rf_models_TRASP_best.pkl"),
        "logs_directory": os.path.exists("logs")
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(),
        "service": "Transport Price Prediction API",
        "files_status": files_status
    }

@app.get("/")
async def root():
    """Endpoint di benvenuto"""
    return {
        "message": "Transport Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)