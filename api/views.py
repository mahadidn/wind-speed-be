# from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
# load model tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # ini akan mengarah ke: prediksi/api
MODEL_DIR = BASE_DIR / "models" / "univariat"
SKALAR_DIR = BASE_DIR / "models" / "skalar"
skalar_path = SKALAR_DIR / "X_scaler.pkl"
isfile = skalar_path.is_file()


MODEL_PATHS_UNIVARIAT = {
    7: load_model( MODEL_DIR / "model_7.keras"),
    15: load_model( MODEL_DIR / "model_15.keras"),
    30: load_model( MODEL_DIR / "model_30.keras"),
    45: load_model( MODEL_DIR / "model_45.keras"),
    60: load_model( MODEL_DIR / "model_60.keras"),
    75: load_model( MODEL_DIR / "model_75.keras"),
    90: load_model( MODEL_DIR / "model_90.keras"),
}



# Create your views here.
# endpoint
@api_view(['GET'])
def get_user(request):
    # return Response({"message": "Hello, World!"})
    return Response({'message': isfile})


@api_view(['POST'])
def prediksi_input_univariat(request):
    try:
        data = request.POST

        # Ambil semua key seperti harike1, harike2, ...
        sorted_keys = sorted(
            [k for k in data.keys() if k.startswith('harike')],
            key=lambda x: int(x.replace('harike', ''))
        )
        input_values = [float(data[k]) for k in sorted_keys]
        jumlah_input = len(input_values)

        # Load dataset
        df = pd.read_csv('https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994-2025-univariat.csv')
        
        # Gunakan hanya fitur 'FF_AVG' untuk univariat
        df = df[['TANGGAL', 'FF_AVG']]
        feature_cols = ['FF_AVG']
        target_col = 'FF_AVG'

        # Split data
        n = len(df)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.20)
        n_test  = n - n_train - n_val

        train_df = df.iloc[:n_train].copy()
        val_df   = df.iloc[n_train : n_train + n_val].copy()
        test_df  = df.iloc[n_train + n_val :].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        # Scaling
        X_scaler = MinMaxScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Siapkan input
        input_array = np.array(input_values).reshape(-1, 1)
        input_scaled = X_scaler.transform(input_array)
        input_seq = np.expand_dims(input_scaled, axis=0)

        print(f"📥 Input scaled shape: {input_seq.shape}")

        # Ambil model sesuai jumlah input
        if jumlah_input not in MODEL_PATHS_UNIVARIAT:
            return Response({
                'error': f"Tidak ada model untuk {jumlah_input} input. Gunakan 7, 15, 30, 45, 60, 75, atau 90 input."
            }, status=status.HTTP_400_BAD_REQUEST)

        model = MODEL_PATHS_UNIVARIAT[jumlah_input]

        # Prediksi
        result = model.predict(input_seq, verbose=0)[0]

        # Denormalisasi hasil prediksi
        result = y_scaler.inverse_transform(result.reshape(-1, 1)).flatten()

        return Response({
            'jumlah_input': jumlah_input,
            'prediction': result.tolist()
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def prediksi_input_dari_file(request):
    try:
        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'File tidak ditemukan'}, status=status.HTTP_400_BAD_REQUEST)

        # Baca file CSV atau Excel
        if file.name.endswith('.csv'):
            df_input = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df_input = pd.read_excel(file)
        else:
            return Response({'error': 'Format file tidak didukung. Gunakan CSV atau Excel.'}, status=status.HTTP_400_BAD_REQUEST)

        # Pastikan kolom 'FF_AVG' tersedia
        if 'FF_AVG' not in df_input.columns:
            return Response({'error': "Kolom 'FF_AVG' tidak ditemukan dalam file."}, status=status.HTTP_400_BAD_REQUEST)

        input_values = df_input['FF_AVG'].dropna().tolist()
        jumlah_input = len(input_values)

        if jumlah_input not in MODEL_PATHS_UNIVARIAT:
            return Response({
                'error': f"Jumlah input tidak valid: {jumlah_input}. Gunakan 7, 15, 30, 45, 60, 75, atau 90 data."
            }, status=status.HTTP_400_BAD_REQUEST)

        # Baca dataset utama
        df = pd.read_csv('https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994-2025-univariat.csv')
        df = df[['TANGGAL', 'FF_AVG']]

        # Split data
        n = len(df)
        n_train = int(n * 0.70)
        train_df = df.iloc[:n_train].copy()

        # Scaling
        X_scaler = MinMaxScaler()
        X_train = train_df[['FF_AVG']].values
        X_scaler.fit(X_train)

        y_scaler = MinMaxScaler()
        y_train = train_df['FF_AVG'].values.reshape(-1, 1)
        y_scaler.fit(y_train)

        # Normalisasi input
        input_array = np.array(input_values).reshape(-1, 1)
        input_scaled = X_scaler.transform(input_array)
        input_seq = np.expand_dims(input_scaled, axis=0)

        # Ambil model yang sesuai
        model = MODEL_PATHS_UNIVARIAT[jumlah_input]

        # Prediksi
        result = model.predict(input_seq, verbose=0)[0]

        # Denormalisasi hasil
        result = y_scaler.inverse_transform(result.reshape(-1, 1)).flatten()

        return Response({
            'jumlah_input': jumlah_input,
            'prediction': result.tolist()
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

