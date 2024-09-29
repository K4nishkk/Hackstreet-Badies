# webcam_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process_image/', views.process_image, name='process_image'),
    path('liveness_detection/', views.liveness_detection, name='liveness_detection'),
    path('upload_document/', views.upload_document, name='upload_document'),
]
