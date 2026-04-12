import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score

from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def validate_model(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)

        validation_results = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
        }

        return validation_results

    except Exception as e:
        logger.error("Error occurred during model validation.", exc_info=True)
        raise CustomException(e, sys)


def cross_validate_model(model, x, y, cv=5):
    try:
        scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

        return {
            "cv_scores": [round(float(score), 4) for score in scores],
            "cv_mean_accuracy": round(float(scores.mean()), 4),
            "cv_std_accuracy": round(float(scores.std()), 4),
        }

    except Exception as e:
        logger.error("Error occurred during cross-validation.", exc_info=True)
        raise CustomException(e, sys)