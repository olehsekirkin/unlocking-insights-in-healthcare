import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

import warnings
import joblib

warnings.filterwarnings('ignore')

# 1. Database Connection Setup

# Replace with your actual database credentials
db_user = "yourusername"
db_password = "yourpassword"
db_host = "yourhost"
db_port = "yourport"
db_name = "healthcare_data"

# Create the connection string
connection_string = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create the SQLAlchemy engine
engine = create_engine(connection_string)

# 2. Data Extraction from MySQL

# Query to extract combined patient data
query = """
SELECT 
    hd.Customer_ID,
    hd.date,
    TIMESTAMPDIFF(YEAR, hd.date, CURDATE()) AS Age,
    hd.children,
    hd.charges,
    hd.Hospital_tier,
    hd.City_tier,
    hd.State_ID,
    me.BMI,
    me.HBA1C,
    CASE WHEN me.Heart_Issues = 'yes' THEN 1 ELSE 0 END AS Heart_Issues,
    CASE WHEN me.Any_Transplants = 'yes' THEN 1 ELSE 0 END AS Any_Transplants,
    CASE WHEN me.Cancer_history = 'yes' THEN 1 ELSE 0 END AS Cancer_history,
    me.NumberOfMajorSurgeries,
    CASE WHEN me.smoker = 'yes' THEN 1 ELSE 0 END AS smoker
FROM 
    Hospitalisation_details hd
JOIN 
    Medical_Examinations me ON hd.Customer_ID = me.Customer_ID;
"""

# Load data into a pandas DataFrame
df = pd.read_sql(query, engine)

# Display first few rows
print("Initial Data Snapshot:")
print(df.head())

# 3. Data Preprocessing

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Handle missing values
# For simplicity, we'll drop rows with missing values.
# Depending on the data, consider imputation or other methods.
df = df.dropna()

# 4. Predicting Hospitalization Charges

print("\n--- Predicting Hospitalization Charges ---")

# Define features and target
features_charges = ['Age', 'children', 'Hospital_tier', 'City_tier',
                    'BMI', 'HBA1C', 'Heart_Issues', 'Any_Transplants',
                    'Cancer_history', 'NumberOfMajorSurgeries', 'smoker']
target_charges = 'charges'

X_charges = df[features_charges]
y_charges = df[target_charges]

# Feature Scaling
scaler_charges = StandardScaler()
X_charges_scaled = scaler_charges.fit_transform(X_charges)

# Split the data
X_train_charges, X_test_charges, y_train_charges, y_test_charges = train_test_split(
    X_charges_scaled, y_charges, test_size=0.2, random_state=42
)

# Initialize Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Hyperparameter Tuning with Grid Search
param_grid_gbr = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_gbr = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid_gbr,
    cv=5,
    n_jobs=-1,
    scoring='r2',
    verbose=1
)

# Fit Grid Search
print("Performing Grid Search for Gradient Boosting Regressor...")
grid_search_gbr.fit(X_train_charges, y_train_charges)

# Best Parameters
print(f"\nBest Parameters for Gradient Boosting Regressor: {grid_search_gbr.best_params_}")

# Best Estimator
best_gbr = grid_search_gbr.best_estimator_

# Predictions
y_pred_charges = best_gbr.predict(X_test_charges)

# Evaluation
mse_charges = mean_squared_error(y_test_charges, y_pred_charges)
r2_charges = r2_score(y_test_charges, y_pred_charges)
print(f"Gradient Boosting Regressor - MSE: {mse_charges:.2f}, R²: {r2_charges:.2f}")

# Feature Importance
feature_importances_charges = pd.Series(best_gbr.feature_importances_, index=features_charges)
feature_importances_charges = feature_importances_charges.sort_values(ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_charges, y=feature_importances_charges.index, palette='viridis')
plt.title('Feature Importances for Hospitalization Charges Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 5. Creating Risk Scores Based on Complications

print("\n--- Creating Risk Scores for Patients Based on Complications ---")

# Define Complication as having Heart Issues or Any Transplants
df['Complication'] = np.where((df['Heart_Issues'] == 1) | (df['Any_Transplants'] == 1), 1, 0)

# Features and target for complication risk
features_risk = ['Age', 'children', 'Hospital_tier', 'City_tier',
                'BMI', 'HBA1C', 'Cancer_history', 'NumberOfMajorSurgeries', 'smoker']
target_risk = 'Complication'

X_risk = df[features_risk]
y_risk = df[target_risk]

# Feature Scaling
scaler_risk = StandardScaler()
X_risk_scaled = scaler_risk.fit_transform(X_risk)

# Split the data with stratification
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_risk_scaled, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)

# Initialize Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)

# Hyperparameter Tuning with Grid Search
param_grid_gbc = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_gbc = GridSearchCV(
    estimator=gbc,
    param_grid=param_grid_gbc,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    verbose=1
)

# Fit Grid Search
print("Performing Grid Search for Gradient Boosting Classifier (Complications)...")
grid_search_gbc.fit(X_train_risk, y_train_risk)

# Best Parameters
print(f"\nBest Parameters for Gradient Boosting Classifier: {grid_search_gbc.best_params_}")

# Best Estimator
best_gbc = grid_search_gbc.best_estimator_

# Predictions
y_pred_risk = best_gbc.predict(X_test_risk)
y_proba_risk = best_gbc.predict_proba(X_test_risk)[:, 1]

# Evaluation
print("\nClassification Report for Complication Risk:")
print(classification_report(y_test_risk, y_pred_risk))
roc_auc_risk = roc_auc_score(y_test_risk, y_proba_risk)
print(f"ROC AUC Score: {roc_auc_risk:.2f}")

# ROC Curve
fpr_risk, tpr_risk, _ = roc_curve(y_test_risk, y_proba_risk)
plt.figure(figsize=(8, 6))
plt.plot(fpr_risk, tpr_risk, label=f'ROC Curve (AUC = {roc_auc_risk:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Complication Risk Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Create Risk Scores Based on Predicted Probabilities
df['Complication_Prob'] = best_gbc.predict_proba(scaler_risk.transform(X_risk))[:, 1]

# Define Risk Categories
def risk_category(prob):
    if prob >= 0.75:
        return 'High Risk'
    elif prob >= 0.5:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

df['Risk_Score'] = df['Complication_Prob'].apply(risk_category)

# Display Risk Score Distribution
print("\nRisk Score Distribution:")
print(df['Risk_Score'].value_counts())

# 6. Summary and Insights

print("\n--- Summary and Insights ---")

# Hospitalization Charges Prediction
print(f"\nHospitalization Charges Prediction - R² Score: {r2_charges:.2f}")
print("Top 3 Features Influencing Charges:")
print(feature_importances_charges.head(3))

# Risk Scores
print("\nRisk Score Distribution:")
print(df['Risk_Score'].value_counts())

# 7. Save Models and Scalers (Optional)

# Save Gradient Boosting Regressor for charges
joblib.dump(best_gbr, 'gradient_boosting_regressor_charges.pkl')
joblib.dump(scaler_charges, 'scaler_charges.pkl')

# Save Gradient Boosting Classifier for complications
joblib.dump(best_gbc, 'gradient_boosting_classifier_complications.pkl')
joblib.dump(scaler_risk, 'scaler_risk.pkl')

print("\nModels and scalers have been saved successfully.")

