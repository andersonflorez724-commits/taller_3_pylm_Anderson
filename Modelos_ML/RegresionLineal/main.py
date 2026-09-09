import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "linear_model.joblib",
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontro el modelo en: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()


class SurfaceInput(BaseModel):
    superficie_m2: float = Field(..., gt=0, description="Superficie de la vivienda en m2")


class PricePrediction(BaseModel):
    superficie_m2: float
    precio_estimado: float


app = FastAPI(
    title="Prediccion de Precios de Viviendas",
    description="Modelo de regresion lineal que estima el precio segun la superficie en m2.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PricePrediction)
def predict(payload: SurfaceInput):
    try:
        x = np.array([[payload.superficie_m2]])
        prediccion = model.predict(x)[0]
        precio = round(float(prediccion), 2)
        return PricePrediction(superficie_m2=payload.superficie_m2, precio_estimado=precio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/precio/{superficie_m2}")
def predict_path(superficie_m2: float):
    if superficie_m2 <= 0:
        raise HTTPException(status_code=400, detail="La superficie debe ser mayor a 0")
    x = np.array([[superficie_m2]])
    prediccion = model.predict(x)[0]
    precio = round(float(prediccion), 2)
    return {"superficie_m2": superficie_m2, "precio_estimado": precio}
