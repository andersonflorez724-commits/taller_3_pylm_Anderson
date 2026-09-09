import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# predecir precios de viviendas según la superficie en M2

# datos de entrenamiento (x) y etiquetas (y)
x = np.array([[40], [50], [60], [85], [100], [150]])
y = np.array([100000, 120000, 150000, 200000, 250000, 300000])

# entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(x, y)

# guardar el modelo entrenado en un archivo
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODELS_DIR, "linear_model.joblib"))
