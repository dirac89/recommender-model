import joblib
from src.common.config import MODEL_PATH

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def make_predictions(model, X):
    # Si es MAPIE v1.x, tiene predict_interval
    if hasattr(model, 'predict_interval'):
        return model.predict_interval(X)
    
    # Fallback para CatBoost / modelos simples
    return model.predict(X)