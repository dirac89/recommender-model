import joblib
import pandas as pd

def load_model(path='models/recommender_model.cbm'):
    return joblib.load(path)

def make_predictions(model, X):
    return model.predict(X)