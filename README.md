# Placement Prediction using Logistic Regression

A simple binary classification project that predicts whether a student will be placed based on their CGPA and IQ score.

## Overview

This project uses a logistic regression model to predict campus placement outcomes from two numerical features: CGPA and IQ. It covers the complete workflow of a basic supervised learning pipeline, including data preprocessing, train-test splitting, feature scaling, model training, evaluation, and visualization of decision boundaries.

## Dataset

The dataset (`placement.csv`) contains 100 student records with the following columns:

| Column      | Description                          |
|-------------|---------------------------------------|
| `cgpa`      | Student's CGPA                        |
| `iq`        | Student's IQ score                    |
| `placement` | Target label (1 = placed, 0 = not placed) |

## Workflow

1. **Preprocessing** — Removed the redundant index column from the raw CSV.
2. **Exploratory Data Analysis** — Visualized the relationship between CGPA, IQ, and placement outcome using a scatter plot.
3. **Train-Test Split** — Split the data into 90% training and 10% testing sets.
4. **Feature Scaling** — Standardized features using `StandardScaler` to bring CGPA and IQ to a comparable scale.
5. **Model Training** — Trained a `LogisticRegression` classifier from scikit-learn on the scaled training data.
6. **Evaluation** — Measured model performance on the test set using accuracy score.
7. **Decision Boundary Visualization** — Plotted the model's decision regions using `mlxtend` to visually inspect the separation between classes.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib
- scikit-learn
- mlxtend

## Project Structure

```
.
├── Placement_Prediction.ipynb   # Main notebook with full implementation
├── placement.csv                 # Dataset
└── README.md
```

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Yash49-Xe/first-ML-project--LogisticRegression-.git
   cd first-ML-project--LogisticRegression-
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn mlxtend
   ```

3. Open and run `Placement_Prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

## Notes

This is an introductory project built to understand the end-to-end logistic regression pipeline, with a focus on preprocessing, scaling, and visualizing decision boundaries rather than extensive hyperparameter tuning or feature engineering.

