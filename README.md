# Diabetes-Prediction-Project
Predict diabetes risk using machine learning models trained on the Pima Indians Diabetes Dataset. This repository provides the full data science workflow—from analysis and preprocessing to model training and export—for reproducible research and easy deployment.

# Project Structure
├── Diabetes_Prediction.ipynb # Main Jupyter notebook for model training & prediction
├── README.md # Project documentation
├── diabetes.csv # Dataset used for training the model
├── diabetes_model.pkl # Trained ML model
├── scaler (2).pkl # Scaler object used for data preprocessing
├── requirements.txt # Python dependencies
├── schema.json # Schema file (possibly for input validation)

# Data Summary
Feature	            Mean	Missing Values
Glucose	           120.89	     5
BloodPressure	     69.11	    35
SkinThickness	     20.54	    227
Insulin	           79.80	    374
BMI	               31.99	    11

# Data Cleaning and Preprocessing
Missing or zero values in certain features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were replaced with the mean value of the respective column.
Feature scaling was applied using a scaler (e.g., StandardScaler) to normalize feature ranges.
Dataset was split into training and test sets with stratified sampling to preserve class balance

# Models Tried
->Logistic Regression
->Random Forest
->Gradient Boosting Classifier
Hyperparameter tuning was performed to optimize model performance. Random Forest provided the best balance of accuracy and robustness.

# Explainability
->SHAP (SHapley Additive exPlanations) was used to interpret feature impact on model predictions.
->Key features impacting diabetes prediction included Glucose, BMI, Age, and Pregnancies.
->Visualization of SHAP values helps in understanding model decisions at both global and individual prediction levels.
