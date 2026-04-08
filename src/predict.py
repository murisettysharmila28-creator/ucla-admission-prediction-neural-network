import joblib
import pandas as pd

from src.config import MODEL_PATH, FEATURE_COLUMNS_PATH
from src.logger import setup_logger

logger = setup_logger()


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info("Model and feature columns loaded successfully.")
        return model, feature_columns
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise


def preprocess_input(data_dict, feature_columns):
    df = pd.DataFrame([data_dict])
    df = df[feature_columns]
    return df


def predict_admission(data_dict):
    model, feature_columns = load_artifacts()

    input_df = preprocess_input(data_dict, feature_columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability