import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def getScalerUni():

    path = Path(__file__).resolve().parent  # -> prediksi/api
    # Buang 'utils' dari path jika ada
    parts = [part for part in path.parts if part != "utils"]
    base_dir = Path(*parts)
    # Gunakan operator / secara aman karena BASE_DIR adalah Path object
    load_data_uni = base_dir / "models" / "datasets" / "1994-2025-univariat.csv"

    # Baca dataset utama
    df = pd.read_csv(load_data_uni)
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

    return X_scaler, y_scaler