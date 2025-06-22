# from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from rest_framework.parsers import MultiPartParser
# load model tensorflow
import pandas as pd
import numpy as np
from api.utils.getScalerUni import getScalerUni
from api.utils.getScalerMulti import getScalerMulti
from api.utils.loadModelsUni import loadModelsUni
from api.utils.loadModelsMulti import loadModelsMulti 
from datetime import datetime, timedelta   


# Create your views here.
# endpoint
@api_view(['GET'])
def get_user(request):
    # return Response({"message": "Hello, World!"})
    return Response({'message': "success"})

# input manual
# univariat
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

        # load scaler
        X_scaler, y_scaler = getScalerUni()

        # Load dataset
        MODEL_PATHS_UNIVARIAT = loadModelsUni()
        if jumlah_input not in MODEL_PATHS_UNIVARIAT:
            return Response({
                'error': f"Tidak ada model untuk {jumlah_input} input. Gunakan 7, 15, 30, 45, 60, 75, atau 90 input."
            }, status=status.HTTP_400_BAD_REQUEST)

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


# multivariat
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

        # load scaler multivariat
        X_scaler, y_scaler = getScalerMulti()
        # Load model multivariat
        MODEL_PATHS_MULTIVARIAT = loadModelsMulti()
        if jumlah_input not in MODEL_PATHS_MULTIVARIAT:
            return Response({
                'error': f"Tidak ada model untuk {jumlah_input} timestep. Gunakan 7, 15, 30, 45, 60, 75, atau 90 timestep."
            }, status=status.HTTP_400_BAD_REQUEST)

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
def prediksi_dari_tanggal_univariat(request):
    try:
        timestep_raw = request.data.get('timestep')
        tanggal_str = request.data.get('tanggalPrediksi')

        if not timestep_raw or not tanggal_str:
            return Response({'error': 'Parameter "timestep" dan "tanggalPrediksi" wajib diisi.'}, status=400)

        try:
            timestep = int(timestep_raw)
        except ValueError:
            return Response({'error': '"timestep" harus berupa angka'}, status=400)

        try:
            target_date = datetime.strptime(tanggal_str, "%d-%m-%Y")
        except ValueError:
            return Response({'error': 'Format "tanggalPrediksi" harus "dd-mm-yyyy".'}, status=400)

        # Load model
        MODEL_PATHS_UNIVARIAT = loadModelsUni()
        if timestep not in MODEL_PATHS_UNIVARIAT:
            return Response({'error': f'Model untuk timestep {timestep} tidak tersedia.'}, status=400)

        # Load data & pastikan format datetime sesuai
        df = pd.read_csv(
            'https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994-2025-univariat.csv'
        )
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%Y-%m-%d')
        df = df[['TANGGAL', 'FF_AVG']]
        df.set_index('TANGGAL', inplace=True)

        # Ambil data untuk input
        input_start = target_date - timedelta(days=timestep)
        input_end = target_date - timedelta(days=1)

        input_df = df.loc[input_start:input_end][['FF_AVG']]
        if len(input_df) != timestep:
            return Response({'error': f'Data tidak mencukupi untuk {timestep} hari sebelum {tanggal_str}.'}, status=400)

        # Ambil data aktual 30 hari ke depan
        pred_start = target_date
        pred_end = target_date + timedelta(days=29)
        actual_df = df.loc[pred_start:pred_end]['FF_AVG']
        tanggal_aktual = [str(t.date()) for t in actual_df.index]

        # Scaling
        X_scaler, y_scaler = getScalerUni()
        input_scaled = X_scaler.transform(input_df.values)

        # Prediksi
        input_seq = np.expand_dims(input_scaled, axis=0)  # shape: (1, timestep, 3)
        model = MODEL_PATHS_UNIVARIAT[timestep]
        result = model.predict(input_seq, verbose=0)[0]
        result = y_scaler.inverse_transform(result.reshape(-1, 1)).flatten()

        return Response({
            'timestep': timestep,
            'tanggal_prediksi': tanggal_str,
            'prediction': result.tolist(),
            'actual': actual_df.tolist() if len(actual_df) == 30 else 'Data aktual tidak lengkap',
            'tipe': 'univariat',
            'tanggal_aktual': tanggal_aktual
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
def prediksi_dari_tanggal_multivariat(request):
    try:
        timestep_raw = request.data.get('timestep')
        tanggal_str = request.data.get('tanggalPrediksi')

        if not timestep_raw or not tanggal_str:
            return Response({'error': 'Parameter "timestep" dan "tanggalPrediksi" wajib diisi.'}, status=400)

        try:
            timestep = int(timestep_raw)
        except ValueError:
            return Response({'error': '"timestep" harus berupa angka'}, status=400)

        try:
            target_date = datetime.strptime(tanggal_str, "%d-%m-%Y")
        except ValueError:
            return Response({'error': 'Format "tanggalPrediksi" harus "dd-mm-yyyy".'}, status=400)

        # Load model
        MODEL_PATHS_MULTIVARIAT = loadModelsMulti()
        if timestep not in MODEL_PATHS_MULTIVARIAT:
            return Response({'error': f'Model untuk timestep {timestep} tidak tersedia.'}, status=400)

        # Load data & pastikan format datetime sesuai
        df = pd.read_csv(
            'https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994_2025_multivariat.csv'
        )
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%Y-%m-%d')
        df = df[['TANGGAL', 'FF_AVG', 'TAVG', 'RH_AVG']]
        df.set_index('TANGGAL', inplace=True)

        # Ambil data untuk input
        input_start = target_date - timedelta(days=timestep)
        input_end = target_date - timedelta(days=1)

        input_df = df.loc[input_start:input_end][['FF_AVG', 'TAVG', 'RH_AVG']]
        if len(input_df) != timestep:
            return Response({'error': f'Data tidak mencukupi untuk {timestep} hari sebelum {tanggal_str}.'}, status=400)

        # Ambil data aktual 30 hari ke depan
        pred_start = target_date
        pred_end = target_date + timedelta(days=29)
        actual_df = df.loc[pred_start:pred_end]['FF_AVG']
        tanggal_aktual = [str(t.date()) for t in actual_df.index]

        # Scaling
        X_scaler, y_scaler = getScalerMulti()
        input_scaled = X_scaler.transform(input_df.values)

        # Prediksi
        input_seq = np.expand_dims(input_scaled, axis=0)  # shape: (1, timestep, 3)
        model = MODEL_PATHS_MULTIVARIAT[timestep]
        result = model.predict(input_seq, verbose=0)[0]
        result = y_scaler.inverse_transform(result.reshape(-1, 1)).flatten()

        return Response({
            'timestep': timestep,
            'tanggal_prediksi': tanggal_str,
            'prediction': result.tolist(),
            'actual': actual_df.tolist() if len(actual_df) == 30 else 'Data aktual tidak lengkap',
            'tipe': 'multivariat',
            'tanggal_aktual': tanggal_aktual
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)


# input file
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

        MODEL_PATHS_UNIVARIAT = loadModelsUni()

        if jumlah_input not in MODEL_PATHS_UNIVARIAT:
            return Response({
                'error': f"Jumlah input tidak valid: {jumlah_input}. Gunakan 7, 15, 30, 45, 60, 75, atau 90 data."
            }, status=status.HTTP_400_BAD_REQUEST)

        # getscaler
        X_scaler, y_scaler = getScalerUni()

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

        MODEL_PATHS_MULTIVARIAT = loadModelsMulti()

        # Ambil jumlah baris input
        timestep = len(df_input)
        if timestep not in MODEL_PATHS_MULTIVARIAT:
            return Response({'error': f'Tidak ada model untuk {timestep} timestep. Gunakan salah satu dari: {list(MODEL_PATHS_MULTIVARIAT.keys())}'}, status=400)

        # Ambil baris terakhir sebanyak timestep
        input_df = df_input[required_cols].tail(timestep).copy()

        X_scaler, y_scaler = getScalerMulti()

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

