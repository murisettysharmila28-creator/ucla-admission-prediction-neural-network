import pandas as pd
from src.logger import setup_logger

logger = setup_logger()


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise