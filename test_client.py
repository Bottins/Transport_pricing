import requests
import json

# URL dell'API (aggiorna con il tuo deployment)
API_URL = "http://localhost:8000/predict"  # Per test locale
# API_URL = "https://your-azure-app.azurewebsites.net/predict"  # Per Azure

# Test per diversi tipi di carico
test_cases = [
    {
        "name": "Test Completo",
        "data": {
            "tipo_carico": "Completo",
            "tipo_trasporto": "1",
            "peso_totale": 28000.0,
            "km_tratta": 918.0,
            "altezza": 260.0,
            "lunghezza_max": 1350.0,
            "misure": 86.0,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "reg_carico": "Veneto",
            "reg_scarico": "Puglia",
            "latitudine_carico": 45.786,
            "longitudine_carico": 12.0,
            "latitudine_scarico": 40.46,
            "longitudine_scarico": 17.26,
            "is_isola": "no",
            "scarico_tassativo": "no",
            "carico_tassativo": "no",
            "data_ordine": "2025-01-15",
            "data_carico": "2025-01-20"
        }
    },
    {
        "name": "Test Parziale",
        "data": {
            "tipo_carico": "Parziale",
            "tipo_trasporto": "1",
            "peso_totale": 5000.0,
            "km_tratta": 450.0,
            "altezza": 180.0,
            "lunghezza_max": 800.0,
            "misure": 15.0,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "reg_carico": "Lombardia",
            "reg_scarico": "Emilia-Romagna",
            "latitudine_carico": 45.4642,
            "longitudine_carico": 9.1900,
            "latitudine_scarico": 44.4949,
            "longitudine_scarico": 11.3426,
            "is_isola": "no",
            "scarico_tassativo": "si",
            "carico_tassativo": "no"
        }
    },
    {
        "name": "Test Groupage",
        "data": {
            "tipo_carico": "Groupage",
            "tipo_trasporto": "1",
            "peso_totale": 350.0,
            "km_tratta": 200.0,
            "altezza": 120.0,
            "lunghezza_max": 120.0,
            "misure": 1.2,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "reg_carico": "Lazio",
            "reg_scarico": "Campania",
            "latitudine_carico": 41.9028,
            "longitudine_carico": 12.4964,
            "latitudine_scarico": 40.8518,
            "longitudine_scarico": 14.2681,
            "is_isola": "no",
            "scarico_tassativo": "no",
            "carico_tassativo": "no"
        }
    }
]

def test_health_check():
    """Test dell'endpoint di health check"""
    try:
        health_url = API_URL.replace("/predict", "/health")
        response = requests.get(health_url, timeout=30)
        
        print("=" * 50)
        print("🏥 HEALTH CHECK")
        print("=" * 50)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Service: {result['service']}")
            print(f"Predictor initialized: {result['predictor_initialized']}")
            print(f"Loaded models: {result['loaded_models']}")
            
            print("\nModels Status:")
            for tipo, status in result['models_status'].items():
                print(f"  {tipo}:")
                for file_type, exists in status.items():
                    status_icon = "✅" if exists else "❌"
                    print(f"    {file_type}: {status_icon}")
        else:
            print(f"❌ Health check failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error in health check: {e}")

def test_predictions():
    """Test delle predizioni per tutti i tipi di carico"""
    
    print("\n" + "=" * 50)
    print("🔮 PREDICTION TESTS")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\n📦 {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                API_URL,
                json=test_case['data'],
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Predizione completata!")
                print(f"💰 Prezzo stimato: €{result['predicted_price']:.2f}")
                print(f"🎯 Confidence: {result['confidence_score']:.1%}")
                print(f"🤖 Tipo carico: {result['tipo_carico']}")
                print(f"📊 Model version: {result['model_version']}")
                print(f"📈 Scale method: {result['scale_method']}")
                
                if result.get('interval_50_lower') and result.get('interval_50_upper'):
                    print(f"📊 Intervallo 50%: €{result['interval_50_lower']:.2f} - €{result['interval_50_upper']:.2f}")
                
                if result.get('prediction_std'):
                    print(f"📏 Std deviation: €{result['prediction_std']:.2f}")
                
                print(f"⚡ Tempo esecuzione: {result['execution_time']:.2f}s")
                
            elif response.status_code == 422:
                print("❌ Errore di validazione:")
                error_detail = response.json().get('detail', 'Unknown validation error')
                print(f"   {error_detail}")
                
            else:
                print(f"❌ Errore: {response.status_code}")
                try:
                    error_detail = response.json().get('detail', response.text)
                    print(f"   {error_detail}")
                except:
                    print(f"   {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Errore di connessione: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ Errore nel parsing JSON: {e}")
            print(f"Risposta raw: {response.text}")
        except Exception as e:
            print(f"❌ Errore generico: {e}")

def test_models_info():
    """Test dell'endpoint models info"""
    try:
        models_url = API_URL.replace("/predict", "/models/info")
        response = requests.get(models_url, timeout=30)
        
        print("\n" + "=" * 50)
        print("ℹ️  MODELS INFO")
        print("=" * 50)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Total models: {result['total_models']}")
            
            for tipo, info in result['models_info'].items():
                print(f"\n📊 {tipo}:")
                print(f"  Version: {info['metadata'].get('version', 'N/A')}")
                print(f"  Features count: {info['features_count']}")
                print(f"  Has GMM: {'✅' if info['has_gmm'] else '❌'}")
                print(f"  Scale method: {info['scale_method']}")
        else:
            print(f"❌ Models info failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error in models info: {e}")

def main():
    """Esegue tutti i test"""
    print("🚀 TESTING TRANSPORT PREDICTION API V2")
    print("=" * 50)
    
    # Test health check
    test_health_check()
    
    # Test models info
    test_models_info()
    
    # Test predictions
    test_predictions()
    
    print("\n" + "=" * 50)
    print("✅ TESTING COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main()