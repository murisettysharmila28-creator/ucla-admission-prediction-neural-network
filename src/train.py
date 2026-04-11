import sys
import joblib
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
from src.validation import validate_model, cross_validate_model, tune_model
from src.custom_exception import CustomException

logger = setup_logger()


def preprocess_target(df):
    try:
        df = df.copy()

        if "Serial_No" in df.columns:
            df = df.drop("Serial_No", axis=1)

        df[TARGET_COLUMN] = (df[TARGET_COLUMN] >= THRESHOLD).astype(int)

        return df

    except Exception as e:
        logger.error("Error occurred during target preprocessing.", exc_info=True)
        raise CustomException(e, sys)


def prepare_data(df):
    try:
        df = preprocess_target(df)

        x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET_COLUMN]

        return x, y

    except Exception as e:
        logger.error("Error occurred while preparing data.", exc_info=True)
        raise CustomException(e, sys)


def split_data(x, y):
    try:
        return train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    except Exception as e:
        logger.error("Error occurred during train-test split.", exc_info=True)
        raise CustomException(e, sys)


def build_preprocessor():
    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        return preprocessor

    except Exception as e:
        logger.error("Error occurred while building preprocessor.", exc_info=True)
        raise CustomException(e, sys)


def build_model():
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
        return model

    except Exception as e:
        logger.error("Error occurred while building baseline model.", exc_info=True)
        raise CustomException(e, sys)


def save_artifacts(model_pipeline, preprocessor, feature_columns):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_pipeline, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

        logger.info(f"Model saved at: {MODEL_PATH}")
        logger.info(f"Preprocessor saved at: {PREPROCESSOR_PATH}")
        logger.info(f"Feature columns saved at: {FEATURE_COLUMNS_PATH}")

    except Exception as e:
        logger.error("Error occurred while saving artifacts.", exc_info=True)
        raise CustomException(e, sys)


def train_and_save_model(df):
    try:
        x, y = prepare_data(df)
        x_train, x_test, y_train, y_test = split_data(x, y)

        preprocessor = build_preprocessor()

        # Baseline model
        baseline_model = build_model()

        baseline_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", baseline_model),
        ])

        baseline_pipeline.fit(x_train, y_train)
        logger.info("Baseline neural network model trained successfully.")

        baseline_train_acc, baseline_test_acc = evaluate_model(
            baseline_pipeline, x_train, y_train, x_test, y_test
        )

        logger.info(f"Baseline Train Accuracy: {baseline_train_acc:.4f}")
        logger.info(f"Baseline Test Accuracy: {baseline_test_acc:.4f}")

        # Tuned model
        tuned_pipeline, best_params, best_cv_score = tune_model(
            preprocessor=preprocessor,
            x_train=x_train,
            y_train=y_train,
            random_state=RANDOM_STATE,
        )

        logger.info(f"Best Parameters: {best_params}")
        logger.info(f"Best CV Score from GridSearch: {best_cv_score:.4f}")

        train_acc, test_acc = evaluate_model(
            tuned_pipeline, x_train, y_train, x_test, y_test
        )

        validation_results = validate_model(tuned_pipeline, x_test, y_test)
        cv_results = cross_validate_model(tuned_pipeline, x, y, cv=5)

        logger.info(f"Tuned Train Accuracy: {train_acc:.4f}")
        logger.info(f"Tuned Test Accuracy: {test_acc:.4f}")
        logger.info(f"Validation Results: {validation_results}")
        logger.info(f"Cross-Validation Results: {cv_results}")

        save_artifacts(
            tuned_pipeline,
            preprocessor,
            x.columns.tolist(),
        )

        return (
            tuned_pipeline,
            train_acc,
            test_acc,
            validation_results,
            cv_results,
            best_params,
            best_cv_score,
            baseline_train_acc,
            baseline_test_acc,
        )

    except Exception as e:
        logger.error("Error occurred during training pipeline.", exc_info=True)
        raise CustomException(e, sys)