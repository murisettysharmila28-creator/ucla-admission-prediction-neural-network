from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_save_model


def main():
    print("Starting admission prediction pipeline...")

    df = load_data(DATA_PATH)

    (
        pipeline,
        train_acc,
        test_acc,
        validation_results,
        cv_results,
        best_params,
        best_cv_score,
        baseline_train_acc,
        baseline_test_acc,
    ) = train_and_save_model(df)

    print("\nTraining completed!")

    print("\nBaseline Model Results:")
    print(f"Baseline Train accuracy: {baseline_train_acc:.4f}")
    print(f"Baseline Test accuracy: {baseline_test_acc:.4f}")

    print("\nTuned Model Results:")
    print(f"Tuned Train accuracy: {train_acc:.4f}")
    print(f"Tuned Test accuracy: {test_acc:.4f}")

    print("\nBest Hyperparameters:")
    print(best_params)
    print(f"Best GridSearch CV Score: {best_cv_score:.4f}")

    print("\nValidation Results:")
    print(f"Accuracy: {validation_results['accuracy']}")
    print(f"Precision: {validation_results['precision']}")
    print(f"Recall: {validation_results['recall']}")
    print(f"F1 Score: {validation_results['f1_score']}")

    print("\nConfusion Matrix:")
    print(validation_results["confusion_matrix"])

    print("\nClassification Report:")
    print(validation_results["classification_report"])

    print("\nCross-Validation Results:")
    print(f"CV Scores: {cv_results['cv_scores']}")
    print(f"Mean Accuracy: {cv_results['cv_mean_accuracy']}")
    print(f"Std Dev: {cv_results['cv_std_accuracy']}")


if __name__ == "__main__":
    main()