from django.urls import path
from .views import get_user, prediksi_input_univariat, prediksi_input_dari_file

urlpatterns = [
    path('users/', get_user, name='get_user'),
    path('prediksi/', prediksi_input_univariat, name='prediksi_view'),
    path('prediksifile/', prediksi_input_dari_file, name='prediksi_input_dari_file'),
]