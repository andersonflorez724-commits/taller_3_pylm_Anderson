"""
URL configuration for frontend project.
"""
from django.urls import path, include

urlpatterns = [
    path('', include('app.urls')),
]
