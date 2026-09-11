import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="API de prediccion de precios de viviendas", description="prediccin de precios de viviendas segun su superficie", version="1.0")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models/linear_model.joblib"

try:
   #cargar el modelo entrenado
   model = joblib.load(MODEL_PATH)
except Exception:
    model = None

#definir el modelo de datos de entrada para la prediccion
class housem2(BaseModel):
    area_m2 : float = Field(...,example=82.5, description="superficie de la vivienda en metros cuadrados")
    
@app.get("/")
def health_check():
    return {"status": "OK", "message": "API de prediccion de precios de viviendas esta en funcionamiento.", "model_loaded": model is not None}

@app.post("/predict")
def predict_price(data: housem2):
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Intente nuevamente más tarde.")
    
    prediction = model.predict([[data.area_m2]])
    
    return {
        "area_m2": data.area_m2,
        "predicted_price": round(float(prediction[0]), 2)
    }           
