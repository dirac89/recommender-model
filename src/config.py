import os

# === Rutas base ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "synthetic_recommendation_dataset.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recommender_model.cbm")
METRICS_PATH = os.path.join(PROJECT_ROOT, "output", "metrics.json")

# === Columnas categóricas para CatBoost ===
CATEGORICAL_COLUMNS = [
    "flight_id",
    "agency_id",
    "route",
    "haul",
    "flight_class",
    "weekday",
    "season",
    "season_extracted",       # NUEVO
    "day_type",               # NUEVO
    "route_cluster"           # NUEVO
]

# === Columna target ===
TARGET_COL = "recommended_seats"

# === Hiperparámetros por defecto para CatBoost ===
DEFAULT_CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "RMSE",
    "verbose": 50
}

# === Configuración de entrenamiento ===
TEST_SIZE = 0.2
RANDOM_STATE = 42
CUTOFF_DATE = "2026-03-01"  # Fecha para split temporal
