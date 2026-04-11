import sys
import joblib
import pandas as pd

from src.config import MODEL_PATH, FEATURE_COLUMNS_PATH
from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def load_prediction_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        logger.info("Prediction artifacts loaded successfully.")
        return model, feature_columns

    except Exception as e:
        logger.error("Error occurred while loading prediction artifacts.", exc_info=True)
        raise CustomException(e, sys)


def predict_admission(input_data: dict):
    try:
        model, feature_columns = load_prediction_artifacts()

        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        logger.info("Prediction generated successfully.")
        return int(prediction), float(probability)

    except Exception as e:
        logger.error("Error occurred during prediction.", exc_info=True)
        raise CustomException(e, sys)