import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Chronic Disease Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("🏥 Chronic Disease Management Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
pages = ["Chronic Disease Overview", "Hospitalization Frequency",
         "Average Charges by Conditions", "BMI Trends", "Readmission Rates",
         "Complications and Outcomes"]
selection = st.sidebar.radio("Go to", pages)

# Database connection
@st.cache_resource
def get_db_connection():
    from sqlalchemy import create_engine

    # Hardcoded database credentials (NOT best practice for anything online, BE AWARE)
    db_user = "yourusername"
    db_password = "yourpassword"
    db_host = "yourhost"
    db_port = "yourport"
    db_name = "healthcare_data"

    connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)
    return engine

engine = get_db_connection()

# Load data from SQL query
@st.cache_data
def load_data(query):
    return pd.read_sql(query, engine)

# 1. Chronic Disease Overview
if selection == "Chronic Disease Overview":
    st.header("📊 Chronic Disease Overview")

    query = """
    SELECT 
        COUNT(*) AS Total_Chronic_Patients,
        AVG(Chronic_Condition_Count) AS Avg_Number_of_Chronic_Conditions
    FROM 
        Chronic_Disease_Patients;
    """
    overview = load_data(query)

    col1, col2 = st.columns(2)
    col1.metric("Total Chronic Patients", int(overview['Total_Chronic_Patients'][0]))
    col2.metric("Average Number of Chronic Conditions", round(overview['Avg_Number_of_Chronic_Conditions'][0], 2))

    st.markdown("---")

# 2. Hospitalization Frequency
elif selection == "Hospitalization Frequency":
    st.header("📈 Hospitalization Frequency for Chronic Disease Patients")

    query = """
    SELECT 
        Customer_ID,
        name,
        COUNT(*) AS Hospitalization_Count,
        MIN(date) AS First_Hospitalization,
        MAX(date) AS Last_Hospitalization,
        DATEDIFF(MAX(date), MIN(date)) AS Days_Between_First_Last
    FROM 
        Chronic_Disease_Patients
    GROUP BY 
        Customer_ID, name
    ORDER BY 
        Hospitalization_Count DESC
    LIMIT 100;
    """
    hosp_freq = load_data(query)

    st.subheader("Top 100 Patients by Hospitalization Count")
    st.dataframe(hosp_freq)

    st.markdown("### Hospitalization Counts Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(hosp_freq['Hospitalization_Count'], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Number of Hospitalizations")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Hospitalization Counts Distribution")
    st.pyplot(fig)

    st.markdown("---")

# 3. Average Charges by Number of Chronic Conditions
elif selection == "Average Charges by Conditions":
    st.header("💰 Average Charges by Number of Chronic Conditions")

    query = """
    SELECT 
        Chronic_Condition_Count,
        AVG(charges) AS Average_Charges,
        SUM(charges) AS Total_Charges,
        COUNT(*) AS Number_of_Patients
    FROM 
        Chronic_Disease_Patients
    GROUP BY 
        Chronic_Condition_Count
    ORDER BY 
        Chronic_Condition_Count;
    """
    charges_data = load_data(query)

    st.subheader("Average Charges and Total Charges by Number of Chronic Conditions")

    st.dataframe(charges_data)

    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x='Chronic_Condition_Count', y='Average_Charges', data=charges_data, palette='viridis', ax=ax1)
    ax1.set_xlabel("Number of Chronic Conditions")
    ax1.set_ylabel("Average Charges ($)")
    ax1.set_title("Average Charges by Number of Chronic Conditions")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.barplot(x='Chronic_Condition_Count', y='Total_Charges', data=charges_data, palette='magma', ax=ax2)
    ax2.set_xlabel("Number of Chronic Conditions")
    ax2.set_ylabel("Total Charges ($)")
    ax2.set_title("Total Charges by Number of Chronic Conditions")
    st.pyplot(fig2)

    st.markdown("---")

# 4. BMI Trends
elif selection == "BMI Trends":
    st.header("📉 BMI Trends Over Time for Chronic Disease Patients")

    query = """
    SELECT 
        Customer_ID,
        YEAR(date) AS Year,
        MONTH(date) AS Month,
        AVG(BMI) AS Average_BMI
    FROM 
        Chronic_Disease_Patients
    GROUP BY 
        Customer_ID, Year, Month
    ORDER BY 
        Customer_ID, Year, Month;
    """
    bmi_trends = load_data(query)

    bmi_agg = bmi_trends.groupby(['Year', 'Month']).agg({'Average_BMI': 'mean'}).reset_index()

    bmi_agg['Date'] = pd.to_datetime(bmi_agg[['Year', 'Month']].assign(DAY=1))

    fig = px.line(bmi_agg, x='Date', y='Average_BMI',
                  title='Average BMI Over Time',
                  labels={'Date': 'Date', 'Average_BMI': 'Average BMI'},
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

# 5. Readmission Rates
elif selection == "Readmission Rates":
    st.header("🔄 Readmission Rates within 30 Days")

    query = """
    SELECT 
        COUNT(*) AS Total_Readmissions,
        AVG(Days_Between) AS Avg_Days_Between
    FROM 
        Readmissions;
    """
    readmission_stats = load_data(query)

    col1, col2 = st.columns(2)
    col1.metric("Total Readmissions within 30 Days", int(readmission_stats['Total_Readmissions'][0]))
    col2.metric("Average Days Between Admissions", round(readmission_stats['Avg_Days_Between'][0], 2))

    query_days = """
    SELECT 
        Days_Between
    FROM 
        Readmissions;
    """
    days_between = load_data(query_days)

    st.subheader("Distribution of Days Between Admissions")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(days_between['Days_Between'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Days Between Admissions")
    ax.set_ylabel("Number of Readmissions")
    ax.set_title("Days Between Readmissions within 30 Days")
    st.pyplot(fig)

    st.markdown("---")

# 6. Complications and Outcomes
elif selection == "Complications and Outcomes":
    st.header("🩺 Complications and Health Outcomes")

    query = """
    SELECT 
        Customer_ID,
        SUM(Has_Heart_Issues) AS Total_Heart_Issues,
        SUM(Has_Diabetes) AS Total_Diabetes,
        SUM(Has_Cancer_History) AS Total_Cancer_History,
        SUM(Has_Transplants) AS Total_Transplants
    FROM 
        Chronic_Disease_Patients
    GROUP BY 
        Customer_ID
    ORDER BY 
        Total_Heart_Issues DESC, Total_Diabetes DESC;
    """
    complications = load_data(query)

    st.subheader("Summary of Complications per Patient")
    st.dataframe(complications)

    complications_agg = complications[['Total_Heart_Issues', 'Total_Diabetes', 'Total_Cancer_History', 'Total_Transplants']].sum().reset_index()
    complications_agg.columns = ['Complication', 'Count']

    st.subheader("Distribution of Complications Among Chronic Disease Patients")
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(complications_agg['Count'], labels=complications_agg['Complication'], autopct='%1.1f%%',
           startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title("Distribution of Complications Among Chronic Disease Patients")
    st.pyplot(fig)

    st.subheader("Total Count of Each Complication")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.barplot(x='Complication', y='Count', data=complications_agg, palette='coolwarm', ax=ax2)
    ax2.set_xlabel("Complications")
    ax2.set_ylabel("Total Count")
    ax2.set_title("Total Count of Each Complication")
    st.pyplot(fig2)

    st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.write("Developed by [Your Name](https://your-website.com)")
