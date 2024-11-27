-- Summary Statistics for Key Metrics (e.g., Average Charges, BMI, HBA1C)

SELECT 
    COUNT(*) AS Total_Patients,
    AVG(charges) AS Average_Charges,
    MIN(charges) AS Minimum_Charges,
    MAX(charges) AS Maximum_Charges,
    STDDEV(charges) AS StdDev_Charges
FROM
    Hospitalisation_details;

-- Average, Minimum, Maximum, and Standard Deviation for BMI
SELECT 
    COUNT(*) AS Total_Patients,
    AVG(BMI) AS Average_BMI,
    MIN(BMI) AS Minimum_BMI,
    MAX(BMI) AS Maximum_BMI,
    STDDEV(BMI) AS StdDev_BMI
FROM
    Medical_Examinations
WHERE
    BMI IS NOT NULL;

-- Average, Minimum, Maximum, and Standard Deviation for HBA1C

SELECT 
    COUNT(*) AS Total_Patients,
    AVG(HBA1C) AS Average_HBA1C,
    MIN(HBA1C) AS Minimum_HBA1C,
    MAX(HBA1C) AS Maximum_HBA1C,
    STDDEV(HBA1C) AS StdDev_HBA1C
FROM
    Medical_Examinations
WHERE
    HBA1C IS NOT NULL;

-- Analyze Distribution of Patients Across Hospital Tiers and City Tiers

SELECT 
    Hospital_tier,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Hospitalisation_details)) * 100,
            2) AS Percentage
FROM
    Hospitalisation_details
GROUP BY Hospital_tier
ORDER BY Hospital_tier;
    
-- Distribution Across City Tiers
SELECT 
    City_tier,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Hospitalisation_details)) * 100,
            2) AS Percentage
FROM
    Hospitalisation_details
GROUP BY City_tier
ORDER BY City_tier;
    
-- Combined Distribution: Hospital Tier vs. City Tier
SELECT 
    h.Hospital_tier,
    h.City_tier,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Hospitalisation_details)) * 100,
            2) AS Percentage
FROM
    Hospitalisation_details h
GROUP BY h.Hospital_tier , h.City_tier
ORDER BY h.Hospital_tier , h.City_tier;

-- Examine the Prevalence of Various Health Conditions

SELECT 
    Heart_Issues,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY Heart_Issues;
    
-- Prevalence of Transplants

SELECT 
    Any_Transplants,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY Any_Transplants;

-- Prevalence of Cancer History

SELECT 
    Cancer_history,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY Cancer_history;

-- Distribution of Number of Major Surgeries
SELECT 
    NumberOfMajorSurgeries,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY NumberOfMajorSurgeries
ORDER BY NumberOfMajorSurgeries;

-- Prevalence of Smoker Status
SELECT 
    smoker,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY smoker;

-- Trend Analysis Over Time, identifying how key metrics and health conditions change over time

SELECT 
    YEAR(date) AS Year,
    MONTH(date) AS Month,
    AVG(charges) AS Average_Charges,
    AVG(BMI) AS Average_BMI,
    COUNT(*) AS Total_Patients
FROM
    Hospitalisation_details h
        JOIN
    Medical_Examinations m ON h.Customer_ID = m.Customer_ID
GROUP BY YEAR(date) , MONTH(date)
ORDER BY Year , Month;

-- Patient Segmentation and Cohort Analysis, segment patients based on various criteria to identify patterns and tailor healthcare services (this one ...
-- ... categorizes patients based on BMI and provides the distribution across these categories)

SELECT 
    CASE
        WHEN BMI < 18.5 THEN 'Underweight'
        WHEN BMI BETWEEN 18.5 AND 24.9 THEN 'Normal'
        WHEN BMI BETWEEN 25 AND 29.9 THEN 'Overweight'
        WHEN BMI >= 30 THEN 'Obese'
        ELSE 'Unknown'
    END AS BMI_Category,
    COUNT(*) AS Patient_Count,
    ROUND((COUNT(*) / (SELECT 
                    COUNT(*)
                FROM
                    Medical_Examinations)) * 100,
            2) AS Percentage
FROM
    Medical_Examinations
GROUP BY BMI_Category
ORDER BY FIELD(BMI_Category,
        'Underweight',
        'Normal',
        'Overweight',
        'Obese',
        'Unknown');
    
-- Cost Analysis and Financioal Insights
SELECT 
    h.Hospital_tier,
    AVG(h.charges) AS Average_Charges,
    SUM(h.charges) AS Total_Charges,
    COUNT(*) AS Number_of_Hospitalizations
FROM
    Hospitalisation_details h
GROUP BY h.Hospital_tier
ORDER BY h.Hospital_tier;

-- Health Outcomes and Treatment Effectiveness
SELECT 
    m.Heart_Issues,
    AVG(m.BMI) AS Average_BMI,
    AVG(m.HBA1C) AS Average_HBA1C,
    AVG(h.charges) AS Average_Charges,
    AVG(m.NumberOfMajorSurgeries) AS Average_Surgeries
FROM
    Medical_Examinations m
        JOIN
    Hospitalisation_details h ON m.Customer_ID = h.Customer_ID
GROUP BY m.Heart_Issues;

-- Geospatial Analysis
SELECT 
    h.State_ID,
    COUNT(*) AS Patient_Count,
    AVG(m.BMI) AS Average_BMI,
    AVG(h.charges) AS Average_Charges
FROM
    Hospitalisation_details h
        JOIN
    Medical_Examinations m ON h.Customer_ID = m.Customer_ID
GROUP BY h.State_ID
ORDER BY Patient_Count DESC;
    
-- Mortality and Morbidity Rates
SELECT 
    h.State_ID,
    COUNT(CASE
        WHEN m.Heart_Issues = 'Yes' THEN 1
    END) AS Heart_Issue_Patients,
    COUNT(CASE
        WHEN m.Cancer_history = 'Yes' THEN 1
    END) AS Cancer_Patients,
    COUNT(*) AS Total_Patients
FROM
    Medical_Examinations m
        JOIN
    Hospitalisation_details h ON m.Customer_ID = h.Customer_ID
GROUP BY h.State_ID;

-- Patient Retention and Repeat Visits
SELECT 
    m.Customer_ID,
    COUNT(*) AS Visit_Count,
    GROUP_CONCAT(DISTINCT DATE_FORMAT(h.date, '%Y-%m')) AS Visit_Months
FROM
    Medical_Examinations m
        JOIN
    Hospitalisation_details h ON m.Customer_ID = h.Customer_ID
GROUP BY m.Customer_ID
HAVING COUNT(*) > 1
ORDER BY Visit_Count DESC;




