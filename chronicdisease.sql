-- 1. Define Chronic Conditions
-- Assuming the following as chronic conditions:
-- - Heart Issues
-- - High HBA1C (Diabetes indicator)
-- - Cancer History
-- - Any Transplants

CREATE OR REPLACE VIEW Chronic_Disease_Patients AS
    SELECT 
        hd.Customer_ID,
        n.name,
        hd.date,
        me.BMI,
        me.HBA1C,
        me.Heart_Issues,
        me.Any_Transplants,
        me.Cancer_history,
        me.NumberOfMajorSurgeries,
        me.smoker,
        hd.charges,
        CASE
            WHEN me.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END AS Has_Heart_Issues,
        CASE
            WHEN me.HBA1C >= 6.5 THEN 1
            ELSE 0
        END AS Has_Diabetes,
        CASE
            WHEN me.Cancer_history = 'yes' THEN 1
            ELSE 0
        END AS Has_Cancer_History,
        CASE
            WHEN me.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END AS Has_Transplants,
        (CASE
            WHEN me.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END + CASE
            WHEN me.HBA1C >= 6.5 THEN 1
            ELSE 0
        END + CASE
            WHEN me.Cancer_history = 'yes' THEN 1
            ELSE 0
        END + CASE
            WHEN me.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END) AS Chronic_Condition_Count
    FROM
        Hospitalisation_details hd
            JOIN
        Medical_Examinations me ON hd.Customer_ID = me.Customer_ID
            JOIN
        Names n ON hd.Customer_ID = n.Customer_ID
    WHERE
        me.Heart_Issues = 'yes'
            OR me.HBA1C >= 6.5
            OR me.Cancer_history = 'yes'
            OR me.Any_Transplants = 'yes';

-- 3. Summary of Chronic Disease Patients
SELECT 
    COUNT(*) AS Total_Chronic_Patients,
    AVG(Chronic_Condition_Count) AS Avg_Number_of_Chronic_Conditions
FROM
    Chronic_Disease_Patients;

-- Analyze Treatment Patterns and Outcomes for Chronic Disease Patients

SELECT 
    Customer_ID,
    COUNT(*) AS Hospitalization_Count,
    MIN(date) AS First_Hospitalization,
    MAX(date) AS Last_Hospitalization,
    DATEDIFF(MAX(date), MIN(date)) AS Days_Between_First_Last
FROM
    Chronic_Disease_Patients
GROUP BY Customer_ID
ORDER BY Hospitalization_Count DESC
LIMIT 100;

-- 2. Average Charges by Number of Chronic Conditions
SELECT 
    Chronic_Condition_Count,
    AVG(charges) AS Average_Charges,
    SUM(charges) AS Total_Charges,
    COUNT(*) AS Number_of_Patients
FROM
    Chronic_Disease_Patients
GROUP BY Chronic_Condition_Count
ORDER BY Chronic_Condition_Count;

-- 3. BMI Trends Over Time for Chronic Disease Patients
SELECT 
    Customer_ID,
    YEAR(date) AS Year,
    MONTH(date) AS Month,
    AVG(BMI) AS Average_BMI
FROM
    Chronic_Disease_Patients
GROUP BY Customer_ID , YEAR(date) , MONTH(date)
ORDER BY Customer_ID , Year , Month;

-- 4. Readmission Rates within 30 Days
CREATE OR REPLACE VIEW Readmissions AS
    SELECT 
        a.Customer_ID,
        a.date AS Admission_Date,
        b.date AS Readmission_Date,
        DATEDIFF(b.date, a.date) AS Days_Between
    FROM
        Chronic_Disease_Patients a
            JOIN
        Chronic_Disease_Patients b ON a.Customer_ID = b.Customer_ID
    WHERE
        b.date > a.date
            AND DATEDIFF(b.date, a.date) <= 30;

-- Readmission Statistics
SELECT 
    COUNT(*) AS Total_Readmissions,
    AVG(Days_Between) AS Avg_Days_Between
FROM
    Readmissions;

-- 5. Complications and Outcomes
SELECT 
    Customer_ID,
    SUM(Has_Heart_Issues) AS Total_Heart_Issues,
    SUM(Has_Diabetes) AS Total_Diabetes,
    SUM(Has_Cancer_History) AS Total_Cancer_History,
    SUM(Has_Transplants) AS Total_Transplants
FROM
    Chronic_Disease_Patients
GROUP BY Customer_ID
ORDER BY Total_Heart_Issues DESC , Total_Diabetes DESC;
