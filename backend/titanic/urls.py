from django.urls import path
from .views import PredictView, HealthCheck

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('', HealthCheck.as_view(), name='health_check'),
]