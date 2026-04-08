from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Admission.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "admission_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

TARGET_COLUMN = "Admit_Chance"

NUMERIC_FEATURES = [
    "GRE_Score",
    "TOEFL_Score",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]

CATEGORICAL_FEATURES = [
    "University_Rating",
]

RANDOM_STATE = 42
THRESHOLD = 0.8