# from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from rest_framework.parsers import MultiPartParser
# load model tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # ini akan mengarah ke: prediksi/api
MODEL_DIR_UNIVARIAT = BASE_DIR / "models" / "univariat"
MODEL_DIR_MULTIVARIAT = BASE_DIR / "models" / "multivariat"
SKALAR_DIR = BASE_DIR / "models" / "skalar"
skalar_path = SKALAR_DIR / "X_scaler.pkl"
isfile = skalar_path.is_file()


import os

import psutil, os
print("RAM usage (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


try:
    print("Model path:", MODEL_DIR_UNIVARIAT / "model_60.keras")
    print("File exists:", os.path.exists(MODEL_DIR_UNIVARIAT / "model_60.keras"))
    model = load_model(MODEL_DIR_UNIVARIAT / "model_60.keras")
except Exception as e:
    print("Gagal load model:", e)


MODEL_PATHS_UNIVARIAT = {
    # 7: load_model( MODEL_DIR_UNIVARIAT / "model_7.keras"),
    # 15: load_model( MODEL_DIR_UNIVARIAT / "model_15.keras"),
    # 30: load_model( MODEL_DIR_UNIVARIAT / "model_30.keras"),
    # 45: load_model( MODEL_DIR_UNIVARIAT / "model_45.keras"),
    60: load_model( MODEL_DIR_UNIVARIAT / "model_60.keras"),
    # 75: load_model( MODEL_DIR_UNIVARIAT / "model_75.keras"),
    # 90: load_model( MODEL_DIR_UNIVARIAT / "model_90.keras"),
}

MODEL_PATHS_MULTIVARIAT = {
    # 7: load_model( MODEL_DIR_MULTIVARIAT / "model_7.keras"),
    # 15: load_model( MODEL_DIR_MULTIVARIAT / "model_15.keras"),
    # 30: load_model( MODEL_DIR_MULTIVARIAT / "model_30.keras"),
    # 45: load_model( MODEL_DIR_MULTIVARIAT / "model_45.keras"),
    60: load_model( MODEL_DIR_MULTIVARIAT / "model_60.keras"),
    # 75: load_model( MODEL_DIR_MULTIVARIAT / "model_75.keras"),
    # 90: load_model( MODEL_DIR_MULTIVARIAT / "model_90.keras"),
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
def prediksi_input_multivariat(request):
    try:
        data = request.POST

        # Ambil semua input yang valid dan urutkan berdasarkan indeksnya
        angin_keys = sorted([k for k in data.keys() if k.startswith('anginke')], key=lambda x: int(x.replace('anginke', '')))
        suhu_keys = sorted([k for k in data.keys() if k.startswith('temperaturke')], key=lambda x: int(x.replace('temperaturke', '')))
        lembap_keys = sorted([k for k in data.keys() if k.startswith('kelembapanke')], key=lambda x: int(x.replace('kelembapanke', '')))

        if not (len(angin_keys) == len(suhu_keys) == len(lembap_keys)):
            return Response({'error': 'Jumlah timestep antar fitur tidak konsisten'}, status=status.HTTP_400_BAD_REQUEST)

        jumlah_input = len(angin_keys)

        # Susun input menjadi array shape (jumlah_input, 3)
        input_values = []
        for i in range(jumlah_input):
            angin = float(data[angin_keys[i]])
            suhu = float(data[suhu_keys[i]])
            lembap = float(data[lembap_keys[i]])
            input_values.append([angin, suhu, lembap])

        input_array = np.array(input_values)  # shape: (jumlah_input, 3)

        # Ambil dataset utama multivariat
        df = pd.read_csv('https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994_2025_multivariat.csv')
        df = df[['TANGGAL', 'FF_AVG', 'TAVG', 'RH_AVG']]

        feature_cols = ['FF_AVG', 'TAVG', 'RH_AVG']
        target_col = 'FF_AVG'

        # Split data
        n = len(df)
        n_train = int(n)
        train_df = df.iloc[:n_train].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        # Scaling
        X_scaler = MinMaxScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        print(X_train_scaled.shape)

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Normalisasi input dan reshape ke (1, timestep, fitur)
        input_scaled = X_scaler.transform(input_array)
        input_seq = np.expand_dims(input_scaled, axis=0)

        print(f"📥 Input shape after scaling: {input_seq.shape}")  # (1, jumlah_input, 3)

        # Ambil model sesuai jumlah timestep
        if jumlah_input not in MODEL_PATHS_MULTIVARIAT:
            return Response({
                'error': f"Tidak ada model untuk {jumlah_input} timestep. Gunakan 7, 15, 30, 45, 60, 75, atau 90 timestep."
            }, status=status.HTTP_400_BAD_REQUEST)

        model = MODEL_PATHS_MULTIVARIAT[jumlah_input]

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




@api_view(['POST'])
@parser_classes([MultiPartParser])
def prediksi_input_multivariat_dari_file(request):
    try:
        file = request.FILES.get('file')

        if file is None:
            return Response({'error': 'File tidak ditemukan'}, status=400)

        # Simpan file sementara
        file_path = default_storage.save(file.name, file)
        full_path = default_storage.path(file_path)

        # Baca file CSV atau Excel
        if file.name.endswith('.csv'):
            df_input = pd.read_csv(full_path)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df_input = pd.read_excel(full_path)
        else:
            return Response({'error': 'Format file tidak didukung (gunakan .csv atau .xlsx)'}, status=400)

        # Kolom yang dibutuhkan
        required_cols = ['FF_AVG', 'TAVG', 'RH_AVG']
        if not all(col in df_input.columns for col in required_cols):
            return Response({'error': f'File harus mengandung kolom: {required_cols}'}, status=400)

        # Ambil jumlah baris input
        timestep = len(df_input)
        if timestep not in MODEL_PATHS_MULTIVARIAT:
            return Response({'error': f'Tidak ada model untuk {timestep} timestep. Gunakan salah satu dari: {list(MODEL_PATHS_MULTIVARIAT.keys())}'}, status=400)

        # Ambil baris terakhir sebanyak timestep
        input_df = df_input[required_cols].tail(timestep).copy()

        # Load dataset utama untuk scaling
        df = pd.read_csv('https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994_2025_multivariat.csv')
        df = df[required_cols]

        # Scaling
        X_scaler = MinMaxScaler()
        X_scaler.fit(df[required_cols])

        y_scaler = MinMaxScaler()
        y_scaler.fit(df[['FF_AVG']])

        # Normalisasi input
        input_scaled = X_scaler.transform(input_df.values)
        input_seq = np.expand_dims(input_scaled, axis=0)

        # Ambil model
        model = MODEL_PATHS_MULTIVARIAT[timestep]

        # Prediksi
        result = model.predict(input_seq, verbose=0)[0]
        result = y_scaler.inverse_transform(result.reshape(-1, 1)).flatten()

        return Response({
            'jumlah_input': timestep,
            'prediction': result.tolist()
        })

    except Exception as e:
        return Response({'error': str(e)}, status=400)

