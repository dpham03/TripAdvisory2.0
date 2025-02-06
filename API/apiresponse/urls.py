from django.urls import path
from . import views

urlpatterns = [
    path('find_airport_path/', views.find_airport_path, name='find_airport_path'),  # API endpoint
]
