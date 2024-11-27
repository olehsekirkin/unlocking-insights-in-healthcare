import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Database connection details
db_user = "yourusername"
db_password = "yourpassword"
db_host = "yourhost"
db_port = "yourport"
db_name = "healthcare_data"

# Create the connection string
connection_string = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Load data from the Cost Analysis View
query = "SELECT * FROM Cost_Analysis_View;"
df_cost = pd.read_sql(query, engine)

# Display first few rows
print(df_cost.head())

# Check for missing values
print(df_cost.isnull().sum())

# Handle missing values
df_cost = df_cost.dropna()

# Verify no missing values remain
print(df_cost.isnull().sum())

# Distribution of Charges
plt.figure(figsize=(10, 6))
sns.histplot(df_cost['charges'], bins=50, kde=True)
plt.title('Distribution of Hospitalization Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Charges by Hospital Tier
plt.figure(figsize=(10, 6))
sns.boxplot(x='Hospital_tier', y='charges', data=df_cost)
plt.title('Hospitalization Charges by Hospital Tier')
plt.xlabel('Hospital Tier')
plt.ylabel('Charges')
plt.show()

# Scatterplot of Charges vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='charges', data=df_cost)
plt.title('Charges vs. Age')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Compute correlation matrix
# Exclude date column before computing the correlation matrix
df_cost_numeric = df_cost.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix for only numeric data
correlation_matrix = df_cost_numeric.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Cost Analysis')
plt.show()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Cost Analysis')
plt.show()

# Define independent variables and dependent variable
X = df_cost[['Age', 'children', 'Hospital_tier', 'City_tier', 'BMI', 'HBA1C',
            'Heart_Issues', 'Any_Transplants', 'Cancer_History', 'NumberOfMajorSurgeries', 'Smoker']]
y = df_cost['charges']

# Add constant term for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())


# Define features and target
features = ['Age', 'children', 'Hospital_tier', 'City_tier', 'BMI', 'HBA1C',
            'Heart_Issues', 'Any_Transplants', 'Cancer_History', 'NumberOfMajorSurgeries', 'Smoker']
target = 'charges'

X = df_cost[features]
y = df_cost[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Random Forest Regressor - MSE: {mse:.2f}, R²: {r2:.2f}")

# Feature Importance
feature_importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importances)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
plt.title('Feature Importances for Healthcare Charges Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
