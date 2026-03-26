import os
import sys
import pandas as pd
import joblib
from datetime import datetime

# Añadir el root al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.generator import generate_inference_data
from src.inference.optimizer import SeatOptimizer
from src.common.config import MODEL_PATH

def run_inference_flow():
    # 1. Configuración de rutas
    inference_dir = "data/inference"
    os.makedirs(inference_dir, exist_ok=True)
    
    # 2. Cargar Modelo y Metadatos
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encontró el modelo en {MODEL_PATH}. Ejecuta 'make pipeline' primero.")
        return

    print("📖 Cargando modelo y metadatos...")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    encoding_map = payload["encoding_map"]
    actual_cat_features = payload["actual_cat_features"]
    
    # 3. Generar Datos SINTÉTICOS para Inferencia
    print("🎲 Generando datos de prueba para un vuelo futuro...")
    df_inference = generate_inference_data(n_agencies=20)
    
    # 4. Preprocesar y Enriquecer
    print("⚙️  Enriqueciendo features (DayType, Season, Clusters)...")
    from src.preprocessing.processor import Preprocessor
    preprocessor = Preprocessor()
    
    # Asegurar formato datetime para enriquecimiento
    df_inference['departure_date'] = pd.to_datetime(df_inference['departure_date'])
    df_inference = preprocessor.enrich_datetime_features(df_inference)
    
    # Para route_cluster, como es una demo, usamos un valor fijo o -1
    # En un sistema real, cargaríamos el mapeo de clusters guardado.
    df_inference['route_cluster'] = "0" 
    
    # Orden exacto esperado por el modelo (derivado de X_train.csv)
    expected_cols = [
        'agency_id', 'days_to_departure', 'route', 'haul', 'flight_class', 
        'weekday', 'season', 'departure_hour', 'international', 'flight_capacity', 
        'load_factor_expected', 'load_factor_lag_3', 'load_factor_lag_7', 
        'agency_rating', 'past_sales', 'past_materialization', 'initial_seats_assigned',
        'season_extracted', 'day_type', 'route_cluster'
    ]
    
    X_inference = df_inference[expected_cols].copy()
    
    X_encoded = X_inference.copy()
    for col in actual_cat_features:
        mapping = encoding_map[col]
        X_encoded[col] = X_inference[col].astype(str).map(mapping).fillna(-1)
    
    # 5. Predicción con Intervalos
    print("🔮 Ejecutando inferencia (MAPIE)...")
    preds, intervals = model.predict_interval(X_encoded)
    
    if intervals.ndim == 3:
        intervals = intervals.reshape(len(intervals), 2)
        
    # 6. Optimización
    print("⚖️  Ejecutando optimización de asientos...")
    CAPACITY = 200 # Capacidad fija para la prueba
    optimizer = SeatOptimizer(capacity_limit=CAPACITY)
    
    # Supongamos prioridades basadas en rating (ya presente en generate_inference_data)
    priorities = df_inference['agency_rating'].values
    
    optimization_result = optimizer.optimize_allocation(
        agency_ids=df_inference['agency_id'].values,
        predicted_demands=preds.ravel(),
        lower_bounds=intervals[:, 0].ravel(),
        upper_bounds=intervals[:, 1].ravel(),
        priorities=priorities
    )
    
    # 7. Guardar Resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(inference_dir, f"predictions_{timestamp}.csv")
    
    optimization_result.to_csv(output_path, index=False)
    print(f"✅ Resultados de inferencia guardados en: {output_path}")
    print("\nResumen de Asignación:")
    print(optimization_result[['agency_id', 'predicted_demand', 'optimized_assignment']].head())
    print(f"Total Asientos Asignados: {optimization_result['optimized_assignment'].sum()} / {CAPACITY}")

if __name__ == "__main__":
    run_inference_flow()
