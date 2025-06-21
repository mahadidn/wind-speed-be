from pathlib import Path
from tensorflow.keras.models import load_model

def loadModelsUni():

    path = Path(__file__).resolve().parent  # -> prediksi/api
    # Buang 'utils' dari path jika ada
    parts = [part for part in path.parts if part != "utils"]
    BASE_DIR = Path(*parts)
    # Gunakan operator / secara aman karena BASE_DIR adalah Path object
    MODEL_DIR_UNIVARIAT = BASE_DIR / "models" / "univariat"

    MODEL_PATHS_UNIVARIAT = {
        7: load_model( MODEL_DIR_UNIVARIAT / "model_7.keras"),
        15: load_model( MODEL_DIR_UNIVARIAT / "model_15.keras"),
        30: load_model( MODEL_DIR_UNIVARIAT / "model_30.keras"),
        45: load_model( MODEL_DIR_UNIVARIAT / "model_45.keras"),
        60: load_model( MODEL_DIR_UNIVARIAT / "model_60.keras"),
        75: load_model( MODEL_DIR_UNIVARIAT / "model_75.keras"),
        90: load_model( MODEL_DIR_UNIVARIAT / "model_90.keras"),
    }

    return MODEL_PATHS_UNIVARIAT