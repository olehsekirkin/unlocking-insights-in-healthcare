import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Database connection details
db_user = "yourusername"
db_password = "yourpassword"
db_host = "yourhost"
db_port = "yourport"
db_name = "healthcare_data"

# Create the connection string
connection_string = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(connection_string)

# Load Aggregated Data by State
aggregated_state = pd.read_sql("SELECT * FROM Aggregated_By_State;", engine)

# Display the first few rows to verify
print(aggregated_state.head())

# If State_IDs are not standard abbreviations, create a mapping
state_mapping = {
    'R1011': 'AL',  # Alabama
    'R1012': 'AK',  # Alaska
    'R1013': 'AZ',  # Arizona
    'R1014': 'AR',  # Arkansas
    'R1015': 'CA',  # California
    'R1016': 'CO',  # Colorado
    'R1017': 'CT',  # Connecticut
    'R1018': 'DE',  # Delaware
    'R1019': 'FL',  # Florida
    'R1020': 'GA',  # Georgia
    'R1021': 'HI',  # Hawaii
    'R1022': 'ID',  # Idaho
    'R1023': 'IL',  # Illinois
    'R1024': 'IN',  # Indiana
    'R1025': 'IA',  # Iowa
    'R1026': 'KS',  # Kansas
}

# Remove trailing characters from 'State_ID'
aggregated_state['State_ID'] = aggregated_state['State_ID'].str.strip()

# Now apply the state mapping again
aggregated_state['State_Abbr'] = aggregated_state['State_ID'].map(state_mapping)

# Verify the mapping
print(aggregated_state[['State_ID', 'State_Abbr']].head())

# Create a choropleth map for Average Charges by State
fig = px.choropleth(
    aggregated_state,
    locations='State_Abbr',        # Column with state abbreviations
    locationmode='USA-states',     # Set to plotly's built-in US states
    color='Average_Charges',       # Metric to visualize
    scope="usa",
    color_continuous_scale="OrRd",
    labels={'Average_Charges': 'Avg Hospitalization Charges'},
    title='Average Hospitalization Charges by State'
)

fig.show()

# Initialize Folium Map centered on the USA
m = folium.Map(location=[37.8, -96], zoom_start=4)

# Define a GeoJSON URL for US states (built-in or hosted online)
# Example GeoJSON source from Plotly
geojson_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'

# Merge aggregated data with GeoJSON
aggregated_state = aggregated_state.rename(columns={'State_Abbr': 'postal'})

# Create a Choropleth layer
folium.Choropleth(
    geo_data=geojson_url,
    name='choropleth',
    data=aggregated_state,
    columns=['postal', 'Average_Charges'],
    key_on='feature.id',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Hospitalization Charges'
).add_to(m)


# Add tooltips
folium.GeoJson(
    geojson_url,
    name='States',
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=['name'],
        aliases=['State:'],
        localize=True
    )
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the interactive map to an HTML file
m.save('folium_average_charges_map.html')

print("Interactive Folium map saved as 'folium_average_charges_map.html'")

import plotly.express as px

# Choropleth Map for Patient Count by State
fig = px.choropleth(
    aggregated_state,
    locations='postal',
    locationmode='USA-states',
    color='Patient_Count',
    scope='usa',
    color_continuous_scale='Blues',
    labels={'Patient_Count': 'Number of Patients'},
    title='Number of Patients by State'
)

fig.show()

# Create a choropleth for Average Charges
fig = go.Figure(data=go.Choropleth(
    locations=aggregated_state['postal'],
    z=aggregated_state['Average_Charges'],
    locationmode='USA-states',
    colorscale='OrRd',
    colorbar_title="Average Charges",
    marker_line_color='white',
    marker_line_width=0.5
))

# Add Scattergeo for Patient Count
fig.add_trace(go.Scattergeo(
    locations=aggregated_state['postal'],
    locationmode='USA-states',
    text=aggregated_state['State_ID'] + '<br>' + aggregated_state['Patient_Count'].astype(str),
    mode='markers',
    marker=dict(
        size=aggregated_state['Patient_Count'] / 100,  # Adjust scaling as needed
        color='blue',
        opacity=0.6,
        line=dict(width=0)
    ),
    name='Patient Count'
))

fig.update_layout(
    title_text='Average Hospitalization Charges and Patient Count by State',
    geo=dict(
        scope='usa',
        projection=go.layout.geo.Projection(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    )
)

fig.show()

detailed_data = pd.read_sql("""
    SELECT 
        h.Customer_ID,
        h.State_ID,
        h.City_tier,
        m.BMI,
        m.HBA1C,
        m.Heart_Issues,
        m.Any_Transplants,
        m.Cancer_history,
        m.smoker
        -- Assuming 'latitude' and 'longitude' columns exist
        -- h.latitude,
        -- h.longitude
    FROM 
        Hospitalisation_details h
    JOIN 
        Medical_Examinations m ON h.Customer_ID = m.Customer_ID;
""", engine)

# Ensure that latitude and longitude are present
if 'latitude' in detailed_data.columns and 'longitude' in detailed_data.columns:
    # Drop rows without geolocation data
    detailed_data = detailed_data.dropna(subset=['latitude', 'longitude'])

    # Initialize Folium Map
    m = folium.Map(location=[37.8, -96], zoom_start=4)

    # Create HeatMap
    heat_data = detailed_data[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_data, radius=8, max_zoom=13).add_to(m)

    # Save Heatmap
    m.save('patient_density_heatmap.html')
    print("Patient density heatmap saved as 'patient_density_heatmap.html'")
else:
    print("Latitude and Longitude data not available. Skipping HeatMap creation.")

# Correlation Analysis
correlation_matrix = aggregated_state[['Average_Charges', 'Average_BMI', 'Average_HBA1C',
                                      'Heart_Issues_Percentage', 'Transplants_Percentage',
                                      'Cancer_History_Percentage', 'Smoker_Percentage']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Health Metrics by State')
plt.show()

# Regression Analysis: Predicting Average Charges
X = aggregated_state[['Average_BMI', 'Average_HBA1C', 'Heart_Issues_Percentage',
                      'Transplants_Percentage', 'Cancer_History_Percentage', 'Smoker_Percentage']]
y = aggregated_state['Average_Charges']

# Add constant term for intercept
X = sm.add_constant(X)

# Fit Ordinary Least Squares (OLS) regression
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Select features for clustering
features = ['Average_Charges', 'Average_BMI', 'Average_HBA1C',
            'Heart_Issues_Percentage', 'Transplants_Percentage',
            'Cancer_History_Percentage', 'Smoker_Percentage']

X_cluster = aggregated_state[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# From the Elbow Curve, assume optimal k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
aggregated_state['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters based on two features
plt.figure(figsize=(10, 7))
sns.scatterplot(data=aggregated_state, x='Average_BMI', y='Average_Charges', hue='Cluster', palette='Set2')
plt.title('Clusters of States based on BMI and Charges')
plt.show()

# Group by 'Cluster' and calculate the mean for numeric columns only
cluster_profiles = aggregated_state.groupby('Cluster').mean(numeric_only=True)

# Print the resulting cluster profiles
print(cluster_profiles)



