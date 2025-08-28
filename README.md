# Diabetes Prediction with Explainable AI
# Project Overview
This project predicts the likelihood of diabetes using the Pima Indians Diabetes Dataset.
We built multiple machine learning models (Logistic Regression, Random Forest, Gradient Boosting) and selected the best one using evaluation metrics.

To ensure trust and transparency in predictions, we applied SHAP (SHapley Additive exPlanations) to interpret model outputs and highlight the most important features affecting diabetes risk.

# Features

Data preprocessing & class imbalance handling

Model training & hyperparameter tuning (GridSearchCV)

Performance evaluation using AUC, Precision, Recall, F1-score

Explainable AI with SHAP values for model interpretation

Visualization of feature importance and SHAP summary plots

# Dataset

We used the Pima Indians Diabetes Dataset from Kaggle/UCI:

Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Target: Outcome (1 = Diabetic, 0 = Non-diabetic)

# Tech Stack

Python 

pandas, numpy (data preprocessing)

scikit-learn (ML models, evaluation)

imbalanced-learn (SMOTE for handling class imbalance)

shap (model explainability)

matplotlib, seaborn (visualization)

# Model Performance (example results)
Model	AUC	Precision	Recall	F1-score
LogisticRegression	0.81	0.603	0.704	0.650
RandomForest	0.82	0.623	0.704	0.661
GradientBoosting	0.82	0.641	0.759	0.695
# Explainability with SHAP

We used SHAP to interpret our Gradient Boosting model:

Glucose, BMI, and Age are the top 3 most important predictors of diabetes.

High Glucose and BMI strongly push predictions towards diabetes (positive SHAP values).

Low values of these features reduce the diabetes risk prediction.

# Example SHAP summary plot:


# How to Run

Clone the repo

git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook or Python script

jupyter notebook diabetes_prediction.ipynb

# Results & Insights

ML models achieved ~82% AUC.

SHAP analysis validated clinical knowledge: Glucose, BMI, and Age are key risk factors.

This project demonstrates how Explainable AI can make healthcare predictions more interpretable and reliable.

📌 Future Work

Try deep learning models (Neural Networks).

Deploy model with a web app (Streamlit/Flask).

Explore additional interpretability methods (LIME, ELI5).
