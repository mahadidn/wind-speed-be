from django.urls import path
from .views import get_user, prediksi_input_univariat, prediksi_input_dari_file, prediksi_input_multivariat

urlpatterns = [
    path('users/', get_user, name='get_user'),
    path('prediksi/univariat', prediksi_input_univariat, name='prediksi_view'),
    path('prediksi/univariat/file', prediksi_input_dari_file, name='prediksi_input_dari_file'),
    path('prediksi/multivariat', prediksi_input_multivariat, name='prediksi_input_multivariat'),
]