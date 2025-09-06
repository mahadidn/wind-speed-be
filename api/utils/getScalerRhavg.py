import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def getScalerRhavg():

    required_cols = ['RH_AVG']

    path = Path(__file__).resolve().parent  # -> prediksi/api
    # Buang 'utils' dari path jika ada
    parts = [part for part in path.parts if part != "utils"]
    base_dir = Path(*parts)
    # Gunakan operator / secara aman karena BASE_DIR adalah Path object
    load_data_multi = base_dir / "models" / "datasets" / "1994_2025_multivariat.csv"

    # Load dataset utama untuk scaling
    df = pd.read_csv(load_data_multi)
    df = df[required_cols]

    y_scaler_rhavg = MinMaxScaler()
    y_scaler_rhavg.fit(df[['RH_AVG']])

    return y_scaler_rhavg