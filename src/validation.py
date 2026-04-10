from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def validate_model(model, x_test, y_test):
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


def cross_validate_model(model, x, y, cv=5):
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    return {
        "cv_scores": [round(float(score), 4) for score in scores],
        "cv_mean_accuracy": round(float(scores.mean()), 4),
        "cv_std_accuracy": round(float(scores.std()), 4),
    }


def tune_model(preprocessor, x_train, y_train, random_state=42):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(random_state=random_state)),
    ])

    param_grid = {
        "classifier__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__solver": ["adam"],
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__learning_rate_init": [0.001, 0.01],
        "classifier__max_iter": [1000, 1500],
        "classifier__early_stopping": [True],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(x_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        round(float(grid_search.best_score_), 4),
    )