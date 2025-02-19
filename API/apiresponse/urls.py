from django.urls import path
from . import views

urlpatterns = [
    path('find_airport_path/', views.find_airport_path, name='find_airport_path'),
    path('find_two_city_path/', views.find_two_city_path, name='find_two_city_path'),
    path('push_image/', views.upload_image, name='push_image'),
    path('push_prompt/', views.upload_prompt, name='push_prompt'),
    path('recommend_cities/', views.recommend_cities, name='recommend_cities')
]
