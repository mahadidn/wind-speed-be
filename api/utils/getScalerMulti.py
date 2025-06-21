import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getScalerMulti():

    required_cols = ['FF_AVG', 'TAVG', 'RH_AVG']

    # Load dataset utama untuk scaling
    df = pd.read_csv('https://raw.githubusercontent.com/mahadidn/wind-speed-forecasting/refs/heads/main/datasets/1994_2025_multivariat.csv')
    df = df[required_cols]

    # Scaling
    X_scaler = MinMaxScaler()
    X_scaler.fit(df[required_cols])

    y_scaler = MinMaxScaler()
    y_scaler.fit(df[['FF_AVG']])

    return X_scaler, y_scaler