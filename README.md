# Drone Delivery On-Time Prediction

## Project Overview
This project presents an end-to-end machine learning pipeline for predicting whether a drone delivery will arrive on time.

## Workflow
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Hyperparameter Optimisation (RandomizedSearchCV)
- Model Comparison
- Bootstrap Validation
- Model Interpretation
- Production Pipeline

## Project Structure
```text
Drone_Delivery_Project/
├── data/
├── notebooks/
│   ├── 01_EDA_Drone_Delivery.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Final_Model.ipynb
├── src/
├── figures/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Models
- Logistic Regression (selected)
- Random Forest
- HistGradient Boosting

## Evaluation Metrics
- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- MCC

## Business Applications
- Last-mile logistics
- Medicine delivery
- Blood transport
- Emergency medical supplies
- Decision-support for drone dispatch

## Technologies
Python, pandas, NumPy, scikit-learn, matplotlib, scipy, joblib

## License
Educational and portfolio purposes.
