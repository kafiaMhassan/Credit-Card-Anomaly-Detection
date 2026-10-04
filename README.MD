# Credit Card Anomaly Detection

## Overview

This project develops a machine learning system for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The project compares supervised and unsupervised approaches and evaluates them using metrics suited to fraud detection.

## Dataset

The dataset contains **284,806 transactions** and 31 columns.

* `Class = 0`: Legitimate transaction
* `Class = 1`: Fraudulent transaction
* `Time`: Seconds elapsed since the first transaction
* `Amount`: Transaction amount
* `V1–V28`: PCA-transformed features

Fraudulent transactions represent approximately **0.18%** of the dataset.

## Models

Four models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* Isolation Forest

Random Forest and XGBoost were tuned using `RandomizedSearchCV`.

## Evaluation

Because the dataset is highly imbalanced, the main evaluation metrics were:

* Precision
* Recall
* F1-score
* PR-AUC

### Final Model

The final model is a **tuned Random Forest** with a classification threshold of **0.34**.

| Metric    | Score |
| --------- | ----: |
| Precision | 0.943 |
| Recall    | 0.837 |
| F1-score  | 0.886 |
| PR-AUC    | 0.866 |

The threshold was adjusted from the default 0.50 to improve the balance between precision and recall.

## Project Structure

```text
Credit Card Anomaly Detection/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── credit_card_anomaly_detection.ipynb
│
├── models/
│   └── random_forest_fraud_model.pkl
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Joblib

## Running the Project

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Key Findings

The supervised tree-based models performed substantially better than Isolation Forest for this dataset.

Random Forest achieved the strongest F1-score after threshold optimization, while XGBoost achieved the highest PR-AUC among the tuned models.

## Future Improvements

* Evaluate the model on more recent transaction data
* Test additional anomaly detection approaches
* Improve the Streamlit interface
* Deploy the application online

```
