import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#predecir precios de viviendas segun la superficie en M2

#datos de entrenamiento con relacion lineal consistente:
#precio = 2000 * m2 + 30000
x = np.array([[30], [40], [50], [60], [70], [80], [85], [90], [100], [110], [120], [130], [140], [150], [160], [170], [180], [190], [200]])

y = np.array([90000, 110000, 130000, 150000, 170000, 190000, 200000, 210000, 230000, 250000, 270000, 290000, 310000, 330000, 350000, 370000, 390000, 410000, 430000])

#entrenar el modelo de regresion lineal con mas datos
model = LinearRegression()
model.fit(x, y)

#predicciones de prueba
y_pred = model.predict(x)

#print la informacion del modelo entrenado
print("Coeficiente de regression: ", model.coef_[0])
print("Termino independiente: ", model.intercept_)

#graficar datos reales
plt.scatter(x, y, color='red', label='Datos reales')

#graficar la linea de regresion
plt.plot(x, y_pred, color='blue', label='Línea de regresión')

plt.xlabel('Superficie (m2)')
plt.ylabel('Precio (COP)')
plt.title('Regresión Lineal: Precio de Viviendas segun Superficie (m2)')
plt.legend()
plt.grid(True)

#imprimir
plt.show()

#guardar el modelo entrenado en un archivo
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/linear_model.joblib')