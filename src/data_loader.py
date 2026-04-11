import sys
import pandas as pd

from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}", exc_info=True)
        raise CustomException(e, sys)