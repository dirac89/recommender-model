import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.generator import generate_synthetic_data
from src.preprocessing.processor import Preprocessor
from src.training.trainer import RecommenderTrainer
from src.inference.optimizer import SeatOptimizer
from src.common.config import RAW_DATA_PATH, PROCESSED_DIR
import pandas as pd
import mlflow

def main():
    mlflow.set_experiment("Flight_Recommender_Revenue_v2")
    with mlflow.start_run(run_name="Level_2_Training"):
        if not os.path.exists(RAW_DATA_PATH):
            print("🚀 Generando datos sintéticos...")
            df = generate_synthetic_data()
            os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
            df.to_csv(RAW_DATA_PATH, index=False)
            print("✅ Datos generados.")

        print("⚙️  Preprocesando datos...")
        preprocessor = Preprocessor()
        # Usamos load_data para asegurar parse_dates
        df_raw = preprocessor.load_data()
        preprocessor.preprocess_and_split(df_raw)

        print("🎯 Entrenando modelo (Nivel 2: Asymmetric + Conformal)...")
        trainer = RecommenderTrainer()
        X_train, X_test, y_train, y_test = trainer.load_data()
        model = trainer.train(X_train, y_train)
        
        print("📈 Evaluando modelo...")
        trainer.evaluate_model(model, X_test, y_test)
        
        print("💾 Guardando modelo...")
        trainer.save_model(model)

        print("\n⚖️  DEMOSTRACIÓN DE OPTIMIZACIÓN (Nivel 2.1)")
        print("------------------------------------------")
        # Tomamos un vuelo específico del set de prueba para demostrar el solver
        X_test_meta = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
        sample_flight_id = X_test_meta['flight_id'].iloc[0]
        flight_data = X_test_meta[X_test_meta['flight_id'] == sample_flight_id]
        
        # Preparamos features para el modelo
        X_sample = flight_data.drop(columns=['flight_id', 'departure_date', 'recommended_seats'] + preprocessor.leaky_features, errors='ignore')
        
        # 🚨 Codificar categorical features usando el encoding_map del trainer
        X_sample_encoded = X_sample.copy()
        for col in trainer.actual_cat_features:
            mapping = trainer.encoding_map[col]
            X_sample_encoded[col] = X_sample[col].astype(str).map(mapping).fillna(-1)
        
        # Predicción con intervalos (MAPIE 1.x API)
        # predict_interval devuelve (preds, intervals) donde intervals es (n_samples, 2, n_alphas)
        # o (n_samples, 2) dependiendo del caso. Forzamos extracción del primer alpha.
        preds, intervals = model.predict_interval(X_sample_encoded)
        
        # Aseguramos que intervals sea 2D (n_samples, 2)
        if intervals.ndim == 3:
            intervals = intervals.reshape(len(intervals), 2)
        
        # Capacidad simulada para este vuelo (ej. 1000 asientos para 636 agencias)
        CAPACITY = 1000
        optimizer = SeatOptimizer(capacity_limit=CAPACITY)
        
        # Supongamos prioridades basadas en 'agency_rating'
        priorities = flight_data['agency_rating'].values
        
        optimization_result = optimizer.optimize_allocation(
            agency_ids=flight_data['agency_id'].values,
            predicted_demands=preds.ravel(),
            lower_bounds=intervals[:, 0].ravel(),
            upper_bounds=intervals[:, 1].ravel(),
            priorities=priorities
        )
        
        print(f"Vuelo: {sample_flight_id} | Capacidad Máxima: {CAPACITY}")
        print(optimization_result[['agency_id', 'predicted_demand', 'conf_lower', 'conf_upper', 'optimized_assignment']])
        print(f"Total Asignado: {optimization_result['optimized_assignment'].sum()}")
        
        print("\n🏁 Pipeline Nivel 2 finalizado exitosamente.")

if __name__ == "__main__":
    main()
