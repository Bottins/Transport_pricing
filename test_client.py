# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 21:52:56 2025

@author: alexq
"""

import requests
import json

# URL dell'API (aggiorna con il tuo deployment)
API_URL = "http://localhost:8000/predict"  # Per test locale
# API_URL = "https://your-azure-app.azurewebsites.net/predict"  # Per Azure

# Test per diversi tipi di carico con nuovi input semplificati
test_cases = [
    {
        "name": "Test Completo - Centinato telonato",
        "data": {
            "tipo_carico": "Completo",
            "tipo_trasporto": "1",  # Merce generica
            "peso_totale": 30000.0,
            "km_tratta": 150.0,
            "altezza": 240.0,
            "lunghezza_max": 0.0,
            "misure": 0,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "latitudine_carico": 45.786,
            "longitudine_carico": 12.0,
            "latitudine_scarico": 40.46,
            "longitudine_scarico": 17.26,
            "tipi_allestimenti": "Base,Centinato telonato",  # Centinato ha precedenza
            "specifiche_allestimento": "base",
            "is_isola": "no",
            "scarico_tassativo": "no",
            "carico_tassativo": "no",
            "data_ordine": "2025-01-15",
            "data_carico": "2025-01-20"
        }
    },
    {
        "name": "Test Parziale - Con sponda idraulica",
        "data": {
            "tipo_carico": "Parziale",
            "tipo_trasporto": 1,  # Può essere numero
            "peso_totale": 10000.0,
            "km_tratta": 1290.0,
            "altezza": 160.0,
            "lunghezza_max": 0.0,
            "misure": 1000,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "latitudine_carico": 45.4642,
            "longitudine_carico": 9.1900,
            "latitudine_scarico": 44.4949,
            "longitudine_scarico": 11.3426,
            "tipi_allestimenti": ["Furgonato in alluminio", "Pianale senza sponde"],  # Prende il primo
            "specifiche_allestimento": "sponda idraulica",
            "is_isola": "no",
            "scarico_tassativo": "si",
            "carico_tassativo": "no"
        }
    },
    {
        "name": "Test Groupage - ADR merce pericolosa",
        "data": {
            "tipo_carico": "Groupage",
            "tipo_trasporto": "5",  # ADR merce pericolosa
            "peso_totale": 1000.0,
            "km_tratta": 760.0,
            "altezza": 100.0,
            "lunghezza_max": 100.0,
            "misure": 12000,
            "naz_carico": "IT",
            "naz_scarico": "IT",
            "latitudine_carico": 44.9028,
            "longitudine_carico": 12.4964,
            "latitudine_scarico": 40.8518,
            "longitudine_scarico": 14.2681,
            "tipi_allestimenti": "Cisternato chimico",
            "specifiche_allestimento": "base",
            "is_isola": "no",
            "scarico_tassativo": "no",
            "carico_tassativo": "no"
        }
    },
    {
        "name": "Test Internazionale - Temperatura negativa",
        "data": {
            "tipo_carico": "Completo",
            "tipo_trasporto": 3,  # Temperatura negativa
            "peso_totale": 22000.0,
            "km_tratta": 1800.0,
            "altezza": 220.0,
            "lunghezza_max": 1360.0,
            "misure": 7500,
            "naz_carico": "IT",
            "naz_scarico": "FR",  # Estero
            "latitudine_carico": 45.4642,
            "longitudine_carico": 9.1900,
            "latitudine_scarico": 48.8566,
            "longitudine_scarico": 2.3522,
            "tipi_allestimenti": "Isotermico con frigo",
            "specifiche_allestimento": "base",
            "is_isola": "si",
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
            print(f"Version: {result.get('version', 'N/A')}")
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
                    print(f"📍 Std deviation: €{result['prediction_std']:.2f}")
                
                # Info aggiuntive calcolate
                if result.get('spazio_calcolato') is not None:
                    print(f"📏 Spazio calcolato: {result['spazio_calcolato']:.2f}")
                
                if result.get('perc_camion') is not None:
                    print(f"🚛 Percentuale camion: {result['perc_camion']:.4f}")
                
                if result.get('tipo_pallet') is not None and result['tipo_pallet'] > 0:
                    pallet_names = {
                        1: "Full Pallet", 2: "Light Pallet", 3: "Ultra Light Pallet",
                        4: "Half Pallet", 5: "Extra Light Pallet", 6: "Quarter Pallet",
                        7: "Mini Quarter", 8: "No match"
                    }
                    print(f"📦 Tipo pallet: {pallet_names.get(result['tipo_pallet'], 'Unknown')}")
                
                if result.get('estero') is not None:
                    print(f"🌍 Trasporto estero: {'Sì' if result['estero'] == 1 else 'No'}")
                
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
            
            # Mostra mapping trasporti disponibili
            if 'transport_mapping' in result:
                print("\n🚚 Transport Types Mapping:")
                for num, name in result['transport_mapping'].items():
                    if name in info.get('supported_transport_types', []):
                        print(f"  {num}: {name} ✅")
        else:
            print(f"❌ Models info failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error in models info: {e}")

def test_root_endpoint():
    """Test dell'endpoint root per vedere i requisiti di input"""
    try:
        root_url = API_URL.replace("/predict", "/")
        response = requests.get(root_url, timeout=30)
        
        print("\n" + "=" * 50)
        print("📋 API ROOT INFO")
        print("=" * 50)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Service: {result['message']}")
            print(f"Version: {result['version']}")
            print(f"Description: {result['description']}")
            
            if 'input_requirements' in result:
                print("\n📥 Input Requirements:")
                print("  Mandatory fields:")
                for field in result['input_requirements']['mandatory']:
                    print(f"    - {field}")
                print("\n  Optional fields:")
                for field in result['input_requirements']['optional']:
                    print(f"    - {field}")
                print("\n  Calculated internally:")
                for field in result['input_requirements']['calculated_internally']:
                    print(f"    - {field}")
        else:
            print(f"❌ Root endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error accessing root: {e}")

def main():
    """Esegue tutti i test"""
    print("🚀 TESTING TRANSPORT PREDICTION API V3")
    print("=" * 50)
    print("Nuovo sistema con gestione semplificata allestimenti")
    print("=" * 50)
    
    # Test health check
    test_health_check()
    
    # Test root endpoint per info
    test_root_endpoint()
    
    # Test models info
    test_models_info()
    
    # Test predictions
    test_predictions()
    
    print("\n" + "=" * 50)
    print("✅ TESTING COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main()