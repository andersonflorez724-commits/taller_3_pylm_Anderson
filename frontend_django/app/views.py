from django.shortcuts import render
from django.conf import settings
import requests


def predict_price(request):
    """
    Vista que maneja el formulario de predicción de precios.
    Envía los datos a la API de FastAPI y devuelve el resultado.
    """
    result = None
    error = None
    area = None

    if request.method == 'POST':
        area = request.POST.get('area_m2', '')

        if area:
            try:
                response = requests.post(
                    f'{settings.API_URL}/predict',
                    json={'area_m2': float(area)},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    result = data.get('predicted_price')
                else:
                    error = f'Error en la API: {response.status_code}'
                    if response.text:
                        error += f' - {response.text}'
            except requests.exceptions.RequestException as e:
                error = f'No se pudo conectar con la API: {str(e)}'
            except Exception as e:
                error = f'Error inesperado: {str(e)}'

    return render(request, 'index.html', {
        'result': result,
        'error': error,
        'area': area,
    })
