-- Time Based Analysis

SELECT 
    YEAR(date) AS Year, COUNT(*) AS Hospitalizations
FROM
    Hospitalisation_details
GROUP BY YEAR(date)
ORDER BY Year;

-- Monthly Hospitalization Counts (Aggregated Over All Years)
SELECT 
    MONTH(date) AS Month, COUNT(*) AS Hospitalizations
FROM
    Hospitalisation_details
GROUP BY MONTH(date)
ORDER BY Month;

-- Seasonal Hospitalization Counts
SELECT 
    CASE
        WHEN MONTH(date) IN (12 , 1, 2) THEN 'Winter'
        WHEN MONTH(date) IN (3 , 4, 5) THEN 'Spring'
        WHEN MONTH(date) IN (6 , 7, 8) THEN 'Summer'
        WHEN MONTH(date) IN (9 , 10, 11) THEN 'Fall'
    END AS Season,
    COUNT(*) AS Hospitalizations
FROM
    Hospitalisation_details
GROUP BY Season
ORDER BY FIELD(Season,
        'Winter',
        'Spring',
        'Summer',
        'Fall');

-- Investigate Patterns In Charges Over Time

SELECT 
    YEAR(date) AS Year, AVG(charges) AS Avg_Charges
FROM
    Hospitalisation_details
GROUP BY YEAR(date)
ORDER BY Year;

-- Monthly Average Charges (Aggregated Over All Years)
SELECT 
    MONTH(date) AS Month, AVG(charges) AS Avg_Charges
FROM
    Hospitalisation_details
GROUP BY MONTH(date)
ORDER BY Month;
    
-- Examine How Health Metrics Change OVer Time for Repeat Patients

SELECT 
    Customer_ID, COUNT(*) AS Hospitalization_Count
FROM
    Hospitalisation_details
GROUP BY Customer_ID
HAVING Hospitalization_Count > 1;
    


