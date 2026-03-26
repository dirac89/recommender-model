import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.common.config import MODEL_PATH, METRICS_PATH
from src.inference.optimizer import SeatOptimizer
import json

app = FastAPI(
    title="Flight Recommender API",
    description="API de Producción para Recomendación y Optimización de Asientos (Level 3)",
    version="1.0.0"
)

# === Global State ===
MODEL_STATE = None

def load_system():
    global MODEL_STATE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run pipeline first.")
    MODEL_STATE = joblib.load(MODEL_PATH)
    print("🚀 Sistema cargado exitosamente.")

@app.on_event("startup")
async def startup_event():
    load_system()

# === Schemas ===
class PredictionRequest(BaseModel):
    agency_id: str
    days_to_departure: int
    route: str
    haul: str
    flight_class: str
    weekday: str
    season: str
    departure_hour: int
    international: int
    flight_capacity: int
    load_factor_expected: float
    load_factor_lag_3: float
    load_factor_lag_7: float
    agency_rating: float
    past_sales: float
    past_materialization: float
    initial_seats_assigned: float
    season_extracted: str
    day_type: str
    route_cluster: str

class PredictionResponse(BaseModel):
    agency_id: str
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    confidence: float

class BatchPredictionRequest(BaseModel):
    flight_id: str
    capacity: int
    agencies: List[PredictionRequest]

class OptimizationResponse(BaseModel):
    flight_id: str
    total_assigned: float
    assignments: List[dict]

# === Endpoints ===

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_STATE is not None}

@app.get("/model/info")
def model_info():
    if not os.path.exists(METRICS_PATH):
        return {"error": "Metrics not found"}
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return {
        "version": "2.0 (Asymmetric + Conformal)",
        "metrics": metrics,
        "confidence_level": MODEL_STATE.get("confidence_level", 0.90) if MODEL_STATE else None
    }

@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(req: PredictionRequest):
    if MODEL_STATE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preparamos el dataframe
    df = pd.DataFrame([req.dict()])
    
    # Codificar categóricas
    cat_cols = MODEL_STATE["actual_cat_features"]
    encoding_map = MODEL_STATE["encoding_map"]
    for col in cat_cols:
        mapping = encoding_map[col]
        df[col] = df[col].astype(str).map(mapping).fillna(-1)
    
    # Predicción
    model = MODEL_STATE["model"]
    preds, intervals = model.predict_interval(df)
    
    # Ravel para asegurar escalares
    p = float(np.asarray(preds).ravel()[0])
    l = float(np.asarray(intervals[:, 0]).ravel()[0])
    u = float(np.asarray(intervals[:, 1]).ravel()[0])
    
    return PredictionResponse(
        agency_id=req.agency_id,
        predicted_demand=p,
        lower_bound=l,
        upper_bound=u,
        confidence=MODEL_STATE["confidence_level"]
    )

@app.post("/predict/batch", response_model=OptimizationResponse)
def predict_batch(req: BatchPredictionRequest):
    if MODEL_STATE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Preparar datos
    df = pd.DataFrame([a.dict() for a in req.agencies])
    
    # 2. Codificar
    cat_cols = MODEL_STATE["actual_cat_features"]
    encoding_map = MODEL_STATE["encoding_map"]
    df_encoded = df.copy()
    for col in cat_cols:
        mapping = encoding_map[col]
        df_encoded[col] = df[col].astype(str).map(mapping).fillna(-1)
    
    # 3. Predicción + Intervalos
    model = MODEL_STATE["model"]
    preds, intervals = model.predict_interval(df_encoded)
    
    # 4. Optimización
    optimizer = SeatOptimizer(capacity_limit=req.capacity)
    # Usamos agency_rating como proxy de prioridad
    priorities = df['agency_rating'].values
    
    opt_df = optimizer.optimize_allocation(
        agency_ids=df['agency_id'].values,
        predicted_demands=preds,
        lower_bounds=intervals[:, 0],
        upper_bounds=intervals[:, 1],
        priorities=priorities
    )
    
    return OptimizationResponse(
        flight_id=req.flight_id,
        total_assigned=float(opt_df['optimized_assignment'].sum()),
        assignments=opt_df.to_dict(orient='records')
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
