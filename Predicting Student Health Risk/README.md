# Predicting Student Health Risk

A multiclass machine-learning project for the Kaggle competition
[Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7/overview).
The objective is to predict a student's `health_condition`—`fit`, `at-risk`,
or `unhealthy`—from lifestyle and wellness data.

## Current results

The notebook trains a class-balanced logistic regression baseline and evaluates
it on a reproducible 80/20 holdout split (`random_state=42`). Its recorded
validation results are:

| Metric | Score |
| --- | ---: |
| Accuracy | 83.37% |
| Macro F1 | 0.71 |
| Weighted F1 | 0.85 |

Class-level performance:

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| `at-risk` | 0.99 | 0.82 | 0.89 |
| `fit` | 0.44 | 0.95 | 0.60 |
| `unhealthy` | 0.46 | 0.93 | 0.62 |

These numbers describe a single local holdout run, not a Kaggle leaderboard
score. Because the classes are heavily imbalanced, accuracy alone can be
misleading; macro F1 and the per-class results provide useful additional
context.

## Dataset

The local `data/` directory contains:

| File | Rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| `train.csv` | 690,088 | 15 | Training features, ID, and target |
| `test.csv` | 295,753 | 14 | Competition test features and ID |
| `sample_submission.csv` | 295,753 | — | Required submission format |

The 13 model features cover sleep, heart rate, BMI, calorie expenditure, step
count, exercise, water intake, diet, stress, physical activity,
smoking/alcohol behavior, and gender. Both numeric and categorical predictors
contain missing values.

The target distribution is imbalanced:

| Class | Share of training data |
| --- | ---: |
| `at-risk` | 85.87% |
| `unhealthy` | 8.36% |
| `fit` | 5.77% |

## Approach

All preprocessing and modeling steps are combined in a scikit-learn
`Pipeline`, helping keep the validation data out of preprocessing decisions.

- The `id` and `health_condition` columns are removed from the model features.
- Numeric missing values are replaced with the training median, then features
  are standardized.
- Categorical missing values are replaced with `unknown`, then values are
  one-hot encoded with unseen categories ignored.
- A logistic regression model is trained with `class_weight="balanced"` and
  `max_iter=1000`.
- Performance is reported with accuracy, a classification report, and a
  confusion matrix.
- The final pipeline is refit on the full training dataset.

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

Use Python 3.10 or newer. From the project directory, create a virtual
environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyter pandas numpy scikit-learn icecream
jupyter notebook main.ipynb
```

Run the notebook from this directory because it loads the CSV files from the
relative `./data/` path. The full dataset is large, so training may take some
time and memory.

## Project status and next steps

Exploration, preprocessing, baseline training, and holdout evaluation are
implemented. Competition submission generation is not complete: the last
prediction cell currently predicts the validation features (`X_test`) instead
of the prepared competition features (`X_test_check`), and it does not write a
submission CSV.

Recommended next steps:

1. Use a stratified train/validation split for a more reliable class
   distribution in each partition.
2. Compare the baseline against nonlinear or boosted-tree classifiers using
   cross-validation and the competition's official metric.
3. Predict `X_test_check`, combine the predictions with `test_ids`, and save
   them in the same column order as `sample_submission.csv`.
4. Record the Kaggle leaderboard result separately from local validation
   metrics.
