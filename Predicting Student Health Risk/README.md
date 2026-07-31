# Predicting Student Health Risk

An exploratory machine-learning project for the Kaggle competition
[Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7/overview).
The goal is to predict a student's `health_condition` from lifestyle and
wellness measurements.

## Project status

The notebook currently loads and inspects the data, checks missing values and
class balance, separates features from the target, identifies numerical and
categorical columns, and begins missing-value handling. Model training,
validation, evaluation, and submission generation are not implemented yet.

## Dataset

The `data/` directory contains:

| File | Rows | Purpose |
| --- | ---: | --- |
| `train.csv` | 690,088 | Training features and the `health_condition` target |
| `test.csv` | 295,753 | Test features used to generate predictions |
| `sample_submission.csv` | 295,753 | Required submission format |

The target classes are `fit`, `at-risk`, and `unhealthy`. Available predictors
include sleep duration and quality, heart rate, BMI, calorie expenditure, step
count, exercise duration, water intake, diet type, stress level, physical
activity level, smoking/alcohol behavior, and gender.

Both training and test data contain missing values. The training target is also
imbalanced, with `at-risk` representing most observations, so a stratified
validation strategy and class-sensitive evaluation are important.

## Project structure

```text
Predicting Student Health Risk/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── main.ipynb
└── README.md
```

## Getting started

Use Python 3.10 or newer. From this directory, create a virtual environment and
install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter pandas numpy
jupyter notebook main.ipynb
```

Run the notebook from the project directory because it reads the CSV files from
the relative `data/` path.

## Suggested next steps

1. Move preprocessing into a reusable pipeline to prevent data leakage.
2. Split the training data with stratification on `health_condition`.
3. Encode categorical features and scale numerical features when required.
4. Establish a simple baseline model and evaluate it with the competition's
   official metric.
5. Train the selected model on all training data and write predictions in the
   same format as `sample_submission.csv`.
