from django.urls import path
from . import views

urlpatterns = [
    path('find_airport_path/', views.find_airport_path, name='find_airport_path'),
    path('find_two_city_path/', views.find_two_city_path, name='find_two_city_path'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('upload_prompt/', views.upload_prompt, name='upload_prompt'),
    path('set_alpha_beta/', views.set_alpha_beta, name='set_alpha_beta'),
    path('find_recommended_cities/', views.find_recommended_cities, name='find_recommended_cities')
]
