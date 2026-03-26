import joblib
import pandas as pd
from src.common.config import MODEL_PATH

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def make_predictions(model, X):
    return model.predict(X)