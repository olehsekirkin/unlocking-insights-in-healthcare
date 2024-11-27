# chronic_disease_management.py

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Database Connection Setup
db_user = "yourusername"
db_password = "yourpassword"
db_host = "yourhost"
db_port = "yourport"
db_name = "healthcare_data"

# Create the connection string
connection_string = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# 2. Data Extraction

# Query to extract chronic disease patients data
query = """
SELECT 
    cdp.Customer_ID,
    cdp.name,
    cdp.date,
    cdp.BMI,
    cdp.HBA1C,
    cdp.Has_Heart_Issues,
    cdp.Has_Diabetes,
    cdp.Has_Cancer_History,
    cdp.Has_Transplants,
    cdp.Chronic_Condition_Count,
    hd.children,
    hd.charges,
    hd.Hospital_tier,
    hd.City_tier,
    hd.State_ID,
    me.NumberOfMajorSurgeries,
    me.smoker
FROM 
    Chronic_Disease_Patients cdp
JOIN 
    Hospitalisation_details hd ON cdp.Customer_ID = hd.Customer_ID
JOIN 
    Medical_Examinations me ON cdp.Customer_ID = me.Customer_ID;
"""

# Load data into a pandas DataFrame
df = pd.read_sql(query, engine)

# Display first few rows
print("Initial Data Snapshot:")
print(df.head())

# 3. Data Preprocessing

# Handle missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

df = df.dropna()

# Feature Engineering
df['Has_Complications'] = np.where(df['Chronic_Condition_Count'] > 1, 1, 0)

# Define Features and Target
features = ['BMI', 'HBA1C', 'Has_Heart_Issues', 'Has_Diabetes',
            'Has_Cancer_History', 'Has_Transplants', 'children',
            'charges', 'Hospital_tier', 'City_tier',
            'NumberOfMajorSurgeries', 'smoker']

target = 'Has_Complications'

X = df[features]
y = df[target]

# Encode Categorical Variables
X['smoker'] = X['smoker'].map({'yes': 1, 'no': 0})

# Define numerical and categorical columns
numeric_features = ['BMI', 'HBA1C', 'children', 'charges',
                    'Hospital_tier', 'City_tier', 'NumberOfMajorSurgeries']
categorical_features = ['smoker']

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='if_binary'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Model Development

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    verbose=2
)

# Fit Grid Search
print("Performing Grid Search for Gradient Boosting Classifier...")
grid_search.fit(X_train, y_train)

# Best Parameters
print(f"\nBest Parameters: {grid_search.best_params_}")

# Best Estimator
best_model = grid_search.best_estimator_

# 5. Model Evaluation

# Predictions
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Complication Prediction')
plt.legend(loc='lower right')
plt.show()

# Feature Importance
importances = best_model.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(best_model.named_steps['preprocessor']
                                       .named_transformers_['cat']
                                       .named_steps['onehot']
                                       .get_feature_names_out(categorical_features))
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10], palette='viridis')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 6. Save the Model and Preprocessor

# Save the trained model
joblib.dump(best_model, 'chronic_disease_complication_model.pkl')
print("Model saved as 'chronic_disease_complication_model.pkl'.")

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved as 'preprocessor.pkl'.")
