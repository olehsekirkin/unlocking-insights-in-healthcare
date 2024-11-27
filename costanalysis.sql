-- Create a view that consolidates all relevant factors influencing charges

CREATE OR REPLACE VIEW Cost_Analysis_View AS
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
        CASE
            WHEN me.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END AS Heart_Issues,
        CASE
            WHEN me.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END AS Any_Transplants,
        CASE
            WHEN me.Cancer_history = 'yes' THEN 1
            ELSE 0
        END AS Cancer_History,
        me.NumberOfMajorSurgeries,
        CASE
            WHEN me.smoker = 'yes' THEN 1
            ELSE 0
        END AS Smoker
    FROM
        Hospitalisation_details hd
            JOIN
        Medical_Examinations me ON hd.Customer_ID = me.Customer_ID;

-- Select all data from the Cost Analysis View
SELECT 
    *
FROM
    Cost_Analysis_View;

-- a. Average Charges by Age Group
SELECT 
    CASE
        WHEN Age < 30 THEN 'Under 30'
        WHEN Age BETWEEN 30 AND 45 THEN '30-45'
        WHEN Age BETWEEN 46 AND 60 THEN '46-60'
        ELSE '61 and above'
    END AS Age_Group,
    AVG(charges) AS Average_Charges,
    COUNT(*) AS Hospitalizations
FROM
    Cost_Analysis_View
GROUP BY Age_Group
ORDER BY CASE
    WHEN Age_Group = 'Under 30' THEN 1
    WHEN Age_Group = '30-45' THEN 2
    WHEN Age_Group = '46-60' THEN 3
    ELSE 4
END;

-- b. Average Charges by BMI Category
SELECT 
    CASE
        WHEN BMI < 18.5 THEN 'Underweight'
        WHEN BMI BETWEEN 18.5 AND 24.9 THEN 'Normal'
        WHEN BMI BETWEEN 25 AND 29.9 THEN 'Overweight'
        WHEN BMI >= 30 THEN 'Obese'
        ELSE 'Unknown'
    END AS BMI_Category,
    AVG(charges) AS Average_Charges,
    COUNT(*) AS Hospitalizations
FROM
    Cost_Analysis_View
GROUP BY BMI_Category
ORDER BY FIELD(BMI_Category,
        'Underweight',
        'Normal',
        'Overweight',
        'Obese',
        'Unknown');

