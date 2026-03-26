import os
import pandas as pd
from catboost import CatBoostRegressor, Pool
from src.config import (
    PROCESSED_DIR,
    MODEL_PATH,
    CATEGORICAL_COLUMNS,
    DEFAULT_CATBOOST_PARAMS,
    TARGET_COL
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
from src.config import METRICS_PATH
import json

class RecommenderTrainer:
    def __init__(self, processed_dir=PROCESSED_DIR, model_path=MODEL_PATH):
        self.processed_dir = processed_dir
        self.model_path = model_path
        self.categorical_columns = CATEGORICAL_COLUMNS

    def load_data(self):
        X_train = pd.read_csv(os.path.join(self.processed_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(self.processed_dir, "X_test.csv"))

        # 🔥 Eliminar columnas no numéricas ni categóricas (como fechas)
        if 'departure_date' in X_train.columns:
            X_train = X_train.drop(columns=['departure_date'])
            X_test = X_test.drop(columns=['departure_date'])

        # Asegurar que las columnas categóricas son strings y no tienen NaNs
        for col in self.categorical_columns:
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna("missing").astype(str)
                X_test[col] = X_test[col].fillna("missing").astype(str)

        y_train = pd.read_csv(os.path.join(self.processed_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(self.processed_dir, "y_test.csv")).values.ravel()
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        train_pool = Pool(X_train, label=y_train, cat_features=self.categorical_columns)
        model = CatBoostRegressor(**DEFAULT_CATBOOST_PARAMS)
        model.fit(train_pool)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        preds = model.predict(X_test)
        
        if root_mean_squared_error:
            rmse = root_mean_squared_error(y_test, preds)
        else:
            rmse = mean_squared_error(y_test, preds, squared=False)
            
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print("📊 Métricas del modelo:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        print(f"✅ Métricas guardadas en: {METRICS_PATH}")
        return metrics

    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save_model(self.model_path)
        print(f"✅ Modelo guardado en: {self.model_path}")

if __name__ == "__main__":
    trainer = RecommenderTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    model = trainer.train(X_train, y_train)
    trainer.evaluate_model(model, X_test, y_test)
    trainer.save_model(model)
