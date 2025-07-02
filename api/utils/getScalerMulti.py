import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def getScalerMulti():

    required_cols = ['FF_AVG', 'TAVG', 'RH_AVG']

    path = Path(__file__).resolve().parent  # -> prediksi/api
    # Buang 'utils' dari path jika ada
    parts = [part for part in path.parts if part != "utils"]
    base_dir = Path(*parts)
    # Gunakan operator / secara aman karena BASE_DIR adalah Path object
    load_data_multi = base_dir / "models" / "datasets" / "1994_2025_multivariat.csv"

    # Load dataset utama untuk scaling
    df = pd.read_csv(load_data_multi)
    df = df[required_cols]

    # Scaling
    X_scaler = MinMaxScaler()
    X_scaler.fit(df[required_cols])

    y_scaler = MinMaxScaler()
    y_scaler.fit(df[['FF_AVG']])

    return X_scaler, y_scaler