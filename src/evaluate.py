import sys
from sklearn.metrics import accuracy_score

from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def evaluate_model(model, x_train, y_train, x_test, y_test):
    try:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        logger.info(f"Train Accuracy: {train_acc:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")

        return train_acc, test_acc

    except Exception as e:
        logger.error("Error occurred during model evaluation.", exc_info=True)
        raise CustomException(e, sys)