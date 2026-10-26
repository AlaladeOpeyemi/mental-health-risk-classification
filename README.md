# Predictive Models for Mental Health Risk Assessment and Classification

DataLab Analytics final project. Machine learning models classify a person as **Healthy**, **Needs Attention** or **Critical** from demographic, occupational, psychological and lifestyle data, with a Streamlit prototype that shows class probabilities and the main factors behind each prediction.

> This is a screening aid built on **synthetic data**. It does not replace clinical diagnosis or professional assessment.

## Results (untouched 20% test set)

| Model | Accuracy | Macro F1 | Recall (Critical) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8222 | 0.8122 | 0.8543 | 0.9499 |
| **Random Forest (selected)** | 0.8263 | 0.8165 | 0.8272 | 0.9497 |
| XGBoost | 0.8281 | 0.8162 | 0.8068 | 0.9509 |

Random Forest was selected on macro F1. The three models are within one percentage point of each other. No Healthy person was ever predicted Critical, or the reverse.

## What the project covers

1. Data quality checks (duplicates, missing values, negatives, outliers)
2. Removal of four columns to prevent target leakage (`Person_ID`, `Burnout_Risk`, `Burnout_Score`, `AI_Wellness_Recommendation`)
3. Encoding, then a stratified 80/20 split **before** imputation, scaling and SMOTE so nothing leaks from the test set
4. Exploratory analysis with interpretation of each chart
5. Feature engineering (for example a combined `Psych_Distress_Index`) and importance-based selection (94 to 42 features)
6. Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation, confusion matrices, ROC curves and variable importance
7. A Streamlit app that returns probabilities and an explanation for each prediction

## Files

- The project notebook (`.ipynb`): the full analysis
- `OpeyemiAlalade_app.py`: Streamlit app
- `requirements.txt`: Python packages
- `OpeyemiAlalade_mental_health_artifacts.joblib`: saved Random Forest model, scaler and column lists used by the app
- The project slides (`.pptx`)

## Run it

```
pip install -r requirements.txt
python -m streamlit run OpeyemiAlalade_app.py
```

The dataset (`mental_health_dataset.csv`, 50,000 synthetic records) is provided by DataLab Analytics. Place it next to the notebook to re-run the analysis.

## Limitations

Synthetic data, so results may not carry over to real populations. About 83% accuracy looks like the ceiling for these features, and Needs Attention is the hardest class to separate. Next steps: validate on real data, calibrate probabilities and check fairness across groups.
