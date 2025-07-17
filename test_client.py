import requests
import json

# URL dell'API (cambia con l'URL del tuo deploy)
API_URL = "https://transport-pricing.onrender.com/predict"  # Esempio per Render
#API_URL = "http://localhost:8000/predict"  # Per test locale

# Dati di test
payload = {
    "data_carico": "2025-07-25",
    "latitudine_carico": 45.786,
    "longitudine_carico": 12.0,
    "latitudine_scarico": 40.46,
    "longitudine_scarico": 17.26,
    "naz_carico": "IT",
    "naz_scarico": "IT",
    "reg_carico": "Veneto",
    "reg_scarico": "Puglia",
    "tipo_trasporto": "1",
    "tipo_carico": "Completo",
    "km_tratta": 918.0,
    "peso_totale": 28000.0,
    "misure": 320000.0,
    "altezza": 100.0,
    "lunghezza_max": 0,
    "prezzo_carb": 1589.5,
    "carico_tassativo": "no",
    "scarico_tassativo": "no",
    "tipi_allestimenti": "Centinato telonato",
    "specifiche_allestimento": "",
    "is_isola": "no",
    "note": "Test API call"
}

try:
    # Effettua la richiesta POST
    response = requests.post(
        API_URL,
        json=payload,  # Usa json invece di data per FastAPI
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Predizione completata!")
        print(f"💰 Prezzo stimato: €{result['predicted_price']:.2f}")
        print(f"🎯 Confidence: {result['confidence_score']:.1%}")
        print(f"🤖 Modello: {result['model_used']}")
        print(f"📊 Range: €{result['price_range_min']:.2f} - €{result['price_range_max']:.2f}")
        print(f"⚡ Variabilità: {result['uncertainty_percentage']:.1f}%")
        print(f"⏱️ Tempo esecuzione: {result['execution_time']:.2f}s")
    else:
        print(f"❌ Errore: {response.status_code}")
        print(response.text)
        
except requests.exceptions.RequestException as e:
    print(f"❌ Errore di connessione: {e}")
except json.JSONDecodeError as e:
    print(f"❌ Errore nel parsing JSON: {e}")
    print(f"Risposta raw: {response.text}")
