# UCLA Admission Prediction using Neural Networks

## Project Overview
This project predicts whether a student has a **high chance of admission (≥ 0.8)** to UCLA using a **Neural Network (MLPClassifier)**.  
The solution was originally developed in a notebook and then transformed into a **modular, production-ready machine learning application** with proper validation, logging, and deployment using Streamlit.

---

## Problem Statement
The goal is to classify applicants into:
- **1 → High chance of admission (≥ 0.8)**
- **0 → Lower chance of admission (< 0.8)**

This converts a regression problem into a **binary classification problem**.

---

## Project Structure
```
ucla-admission-prediction-neural-network/
│
├── app.py                      # Streamlit app
├── main.py                     # Training pipeline entry point
├── requirements.txt
├── README.md
│
├── data/
│   └── Admission.csv
│
├── model/
│   ├── admission_model.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
│
├── logs/
│   └── app.log                 # Logging output
│
├── src/
│   ├── config.py
│   ├── custom_exception.py     # Custom error handling
│   ├── logger.py               # Logging setup
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validation.py
│   └── predict.py
│
└── notebooks/
    └── UCLA_Neural_Networks_Solution.ipynb
```

---

## Code Modularization Approach
The project follows a **modular architecture** to ensure reusability and maintainability:

- `data_loader.py` - Loads dataset with error handling
- `train.py` - Handles preprocessing, training, tuning
- `evaluate.py` - Computes accuracy metrics
- `validation.py` - Cross-validation + detailed metrics
- `predict.py` - Inference logic for Streamlit
- `logger.py` - Centralized logging system
- `custom_exception.py` - Structured error handling

This separation ensures clean pipelines and production-level structure.

---

## Modelling Approach

### Preprocessing
- StandardScaler - Numeric features
- OneHotEncoder - Categorical features
- Target transformation:
```
Admit_Chance ≥ 0.8 → 1
Admit_Chance < 0.8 → 0
```

### Model Used
- **MLPClassifier (Neural Network)**
- Hidden layers: `(32, 16)`
- Activation: `ReLU`
- Early stopping enabled

---

## Training Results

| Model Type        | Train Accuracy | Test Accuracy |
|------------------|--------------|--------------|
| Baseline Model   | ~92%         | ~82%         |
| Tuned Model      | ~95%         | **90%**      |

---

## Validation Results

### Hold-Out Validation
- Test Accuracy: **90%**

### Cross-Validation (5-Fold)
- Mean Accuracy: **~91%**
- Stable performance across folds

### Detailed Metrics
- Precision: ~0.80  
- Recall: ~0.90  
- F1 Score: ~0.85  

Validation confirms that the tuned model **generalizes well and is not overfitting**.

---

## Interpretation
- Neural network captures **non-linear relationships** between academic features.
- Hyperparameter tuning significantly improved performance.
- Cross-validation ensured **robust and reliable results**.
- Model is suitable for **decision-support scenarios**, not real admissions.

---

## Streamlit App
The model is deployed using Streamlit for interactive predictions.

### Run the App
```bash
pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

### Features
- User inputs academic scores
- Model predicts admission likelihood
- Displays prediction + probability

---

## Logging & Error Handling
- Logs stored in `logs/app.log`
- Captures:
  - dataset loading
  - training progress
  - evaluation metrics
  - errors with stack trace

- Custom exception handling ensures:
  - traceable errors
  - robust execution

---

## Key Findings
- Neural networks can perform well on structured tabular data when tuned.
- Validation is critical - single test split can be misleading.
- Modular design improves scalability and deployment readiness.

---

## Limitations
- Dataset size is relatively small
- Model may not generalize to real-world admissions
- No external features (e.g., extracurriculars, essays)

---

## Challenges
- Converting notebook into modular pipeline
- Ensuring consistent preprocessing during inference
- Handling overfitting in neural networks
- Implementing proper validation

---

## Learning Outcomes
- Built end-to-end ML pipeline
- Learned model validation techniques
- Applied modular software design in ML
- Implemented logging and error handling
- Deployed ML app using Streamlit

---

## Future Enhancements
- Add SHAP for model explainability
- Deploy on cloud (Streamlit Cloud / AWS)
- Improve feature engineering
- Add real-time prediction API

---

## Author

Sharmila Murisetty  
Data Analyst / Business Intelligence Developer
