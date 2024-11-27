import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Replace with your actual database credentials
db_user = "yourusername"
db_password = "yourpassword"
db_host = "yourhost"
db_port = "yourport"
db_name = "healthcare_data"

# Create the connection string
connection_string = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Query the Patient_Segmentation_View
query = "SELECT * FROM Patient_Segmentation_View;"
df = pd.read_sql(query, engine)

# Select features for clustering
features = ['Age', 'children', 'charges', 'Hospital_tier', 'City_tier',
            'BMI', 'HBA1C', 'Heart_Issues', 'Any_Transplants',
            'Cancer_history', 'NumberOfMajorSurgeries', 'smoker']

X = df[features]

# Handle missing values if any
X = X.dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Fit K-means with k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Summary Statistics by Cluster
cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
print(cluster_summary)

# Count of Patients in Each Cluster
cluster_counts = df['Cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Patient_Count']
print(cluster_counts)

# Reduce dimensions to 2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
df['PC1'] = principal_components[:, 0]
df['PC2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
plt.title('Patient Segments Visualization')
plt.show()

# Merge cluster information with original data
cluster_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'children': 'mean',
    'charges': 'mean',
    'Hospital_tier': 'mean',
    'City_tier': 'mean',
    'BMI': 'mean',
    'HBA1C': 'mean',
    'Heart_Issues': 'mean',
    'Any_Transplants': 'mean',
    'Cancer_history': 'mean',
    'NumberOfMajorSurgeries': 'mean',
    'smoker': 'mean'
}).reset_index()

print(cluster_profiles)

# Heatmap of cluster profiles
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_profiles.set_index('Cluster'), annot=True, cmap='YlGnBu')
plt.title('Cluster Profiles')
plt.show()
