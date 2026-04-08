import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_COLUMNS_PATH,
    RANDOM_STATE,
    THRESHOLD,
)
from src.evaluate import evaluate_model
from src.logger import setup_logger

logger = setup_logger()


def preprocess_target(df):
    df = df.copy()

    if "Serial_No" in df.columns:
        df = df.drop("Serial_No", axis=1)

    df[TARGET_COLUMN] = (df[TARGET_COLUMN] >= THRESHOLD).astype(int)

    return df


def prepare_data(df):
    df = preprocess_target(df)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    return x, y


def split_data(x, y):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_model():
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return model


def save_artifacts(model_pipeline, preprocessor, feature_columns):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    logger.info(f"Model saved at: {MODEL_PATH}")
    logger.info(f"Preprocessor saved at: {PREPROCESSOR_PATH}")
    logger.info(f"Feature columns saved at: {FEATURE_COLUMNS_PATH}")


def train_and_save_model(df):
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    preprocessor = build_preprocessor()
    model = build_model()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(x_train, y_train)
    logger.info("Neural network model trained successfully.")

    train_acc, test_acc = evaluate_model(
        pipeline, x_train, y_train, x_test, y_test
    )

    save_artifacts(
        pipeline,
        preprocessor,
        x.columns.tolist()
    )

    return pipeline, train_acc, test_acc