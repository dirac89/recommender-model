import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    
    Interpretación:
    - PSI < 0.1: No hay cambio significativo (Estable)
    - 0.1 <= PSI < 0.25: Cambio moderado (Monitorear)
    - PSI >= 0.25: Gran cambio (Retrain / Investigar)
    """
    def scale_range(input_array, min_val, max_val):
        return (input_array - min_val) / (max_val - min_val + 1e-10)

    # 1. Definir buckets basados en la población esperada (training)
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 // buckets))
    breakpoints = np.unique(breakpoints) # Evitar buckets duplicados
    
    def bucketize(array, breakpoints):
        # Asignar cada valor a un bucket
        return np.digitize(array, breakpoints)

    expected_percents = (pd.Series(bucketize(expected, breakpoints)).value_counts(normalize=True).sort_index())
    actual_percents = (pd.Series(bucketize(actual, breakpoints)).value_counts(normalize=True).sort_index())

    # Asegurar que ambos tengan los mismos índices (rellenar con epsilon)
    all_buckets = sorted(list(set(expected_percents.index) | set(actual_percents.index)))
    expected_percents = expected_percents.reindex(all_buckets, fill_value=1e-5)
    actual_percents = actual_percents.reindex(all_buckets, fill_value=1e-5)

    # 2. Calcular PSI
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

class DriftMonitor:
    def __init__(self, training_data_path):
        self.training_df = pd.read_csv(training_data_path)
        print(f"📡 Monitor de Drift cargado con datos de entrenamiento ({len(self.training_df)} filas)")

    def check_drift(self, current_df, features=['days_to_departure', 'agency_rating', 'past_materialization']):
        results = {}
        for feature in features:
            if feature not in current_df.columns or feature not in self.training_df.columns:
                continue
            
            psi = calculate_psi(self.training_df[feature], current_df[feature])
            status = "STABLE"
            if psi >= 0.25: status = "CRITICAL DRIFT"
            elif psi >= 0.1: status = "MODERATE DRIFT"
            
            results[feature] = {
                "psi": round(psi, 4),
                "status": status
            }
        return results

if __name__ == "__main__":
    # Demo de uso
    import os
    from src.common.config import PROCESSED_DIR
    
    train_path = os.path.join(PROCESSED_DIR, "X_train.csv")
    if os.path.exists(train_path):
        monitor = DriftMonitor(train_path)
        
        # Simular datos actuales (sin drift)
        X_train = pd.read_csv(train_path)
        current_no_drift = X_train.sample(5000)
        
        # Simular datos con drift (desplazamiento en agency_rating)
        current_with_drift = current_no_drift.copy()
        current_with_drift['agency_rating'] = current_with_drift['agency_rating'] + 0.4 
        
        print("\n🔍 Verificación SIN drift:")
        print(monitor.check_drift(current_no_drift))
        
        print("\n🔥 Verificación CON drift:")
        print(monitor.check_drift(current_with_drift))
