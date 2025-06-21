import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getScalerUni():
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

    return X_scaler, y_scaler