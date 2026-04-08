from src.config import DATA_PATH
from src.data_loader import load_data
from src.train import train_and_save_model


def main():
    print("Starting admission prediction pipeline...")

    df = load_data(DATA_PATH)
    _, train_acc, test_acc = train_and_save_model(df)

    print("\nTraining completed!")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()