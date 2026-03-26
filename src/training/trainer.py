import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.common.config import (
    PROCESSED_DIR,
    MODEL_PATH,
    METRICS_PATH,
    CATEGORICAL_COLUMNS,
    DEFAULT_CATBOOST_PARAMS,
    ALPHA_OVER_ALLOCATION,
    CONFIDENCE_LEVEL
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
import json
import mlflow
import mlflow.catboost

# === Modelo de Recomendación de Asientos (Nivel 2) ===

class RecommenderTrainer:
    def __init__(self, processed_dir=PROCESSED_DIR, model_path=MODEL_PATH):
        self.processed_dir = processed_dir
        self.model_path = model_path
        self.categorical_columns = CATEGORICAL_COLUMNS
        self.alpha = ALPHA_OVER_ALLOCATION
        self.confidence = CONFIDENCE_LEVEL

    def load_data(self):
        X_train = pd.read_csv(os.path.join(self.processed_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(self.processed_dir, "X_test.csv"))

        # Columnas a eliminar (IDs y fechas que no son features)
        drop_cols = ['departure_date', 'flight_id']
        X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

        # Actualizar lista de categóricas basada en lo que queda en el DF
        self.actual_cat_features = [c for c in self.categorical_columns if c in X_train.columns]

        # Asegurar string para cat_features
        for col in self.actual_cat_features:
            X_train[col] = X_train[col].fillna("missing").astype(str)
            X_test[col] = X_test[col].fillna("missing").astype(str)

        y_train = pd.read_csv(os.path.join(self.processed_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.processed_dir, "y_test.csv")).values.ravel()
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print(f"🎯 Entrenando modelo con pérdida asimétrica (Quantile tau=0.4)...")
        
        # 1. Crear mapeo manual para categóricas (más estable que OrdinalEncoder)
        self.encoding_map = {}
        X_train_encoded = X_train.copy()
        
        for col in self.actual_cat_features:
            unique_vals = X_train[col].unique()
            self.encoding_map[col] = {str(val): i for i, val in enumerate(unique_vals)}
            # Aplicar mapeo
            X_train_encoded[col] = X_train[col].astype(str).map(self.encoding_map[col]).fillna(-1)

        # 2. Configurar CatBoost
        params = DEFAULT_CATBOOST_PARAMS.copy()
        params.update({
            "iterations": 300,
            "learning_rate": 0.1,
            "loss_function": "Quantile:alpha=0.4",
            "eval_metric": "Quantile:alpha=0.4",
            "verbose": 50
        })
        
        # MLflow: Log Parameters
        if mlflow.active_run():
            mlflow.log_params(params)
            mlflow.log_param("confidence_level", self.confidence)
            mlflow.log_param("alpha_asymmetry", self.alpha)

        base_model = CatBoostRegressor(**params)
        
        # 3. Usar SplitConformalRegressor
        from mapie.regression import SplitConformalRegressor
        from sklearn.model_selection import train_test_split
        
        X_tr, X_cal, y_tr, y_cal = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=42)
        
        mapie = SplitConformalRegressor(base_model, confidence_level=self.confidence, prefit=False)
        mapie.fit(X_tr, y_tr)
        mapie.conformalize(X_cal, y_cal)
        
        return mapie
    
    def evaluate_model(self, model, X_test, y_test):
        # Sampling
        if len(X_test) > 50000:
            indices = np.random.choice(len(X_test), 50000, replace=False)
            X_test_eval = X_test.iloc[indices].copy()
            y_test_eval = y_test[indices]
        else:
            X_test_eval = X_test.copy()
            y_test_eval = y_test

        # Codificar usando el mapa manual
        X_test_encoded = X_test_eval.copy()
        for col in self.actual_cat_features:
            mapping = self.encoding_map[col]
            X_test_encoded[col] = X_test_eval[col].astype(str).map(mapping).fillna(-1)
        
        # predict_interval
        preds, intervals = model.predict_interval(X_test_encoded)
        
        # Asegurar 1D para cálculos de métricas
        preds = np.asarray(preds).ravel()
        lower_bounds = np.asarray(intervals[:, 0]).ravel()
        upper_bounds = np.asarray(intervals[:, 1]).ravel()
        
        if root_mean_squared_error:
            rmse = root_mean_squared_error(y_test_eval, preds)
        else:
            rmse = mean_squared_error(y_test_eval, preds, squared=False)
            
        mae = mean_absolute_error(y_test_eval, preds)
        r2 = r2_score(y_test_eval, preds)
        
        coverage = ((y_test_eval >= lower_bounds) & (y_test_eval <= upper_bounds)).mean()
        avg_width = (upper_bounds - lower_bounds).mean()

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "coverage_90": coverage,
            "avg_interval_width": avg_width
        }

        # MLflow: Log Metrics
        if mlflow.active_run():
            mlflow.log_metrics(metrics)

        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n📊 Métricas del modelo (Nivel 2):")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        return metrics

    def save_model(self, model):
        # NOTA: Guardamos el modelo MAPIE + Metadata necesaria para inferencia
        import joblib
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        payload = {
            "model": model,
            "encoding_map": self.encoding_map,
            "actual_cat_features": self.actual_cat_features,
            "confidence_level": self.confidence,
            "alpha": self.alpha
        }
        
        joblib.dump(payload, self.model_path)
        
        # MLflow: Log Artifact
        if mlflow.active_run():
            mlflow.log_artifact(self.model_path)
            
        print(f"✅ Modelo + Metadata (Nivel 2) guardado en: {self.model_path}")

if __name__ == "__main__":
    trainer = RecommenderTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    model = trainer.train(X_train, y_train)
    trainer.evaluate_model(model, X_test, y_test)
    trainer.save_model(model)
