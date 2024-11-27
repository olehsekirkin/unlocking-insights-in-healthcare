-- Aggregated Metrics by State

CREATE OR REPLACE VIEW Aggregated_By_State AS
    SELECT 
        h.State_ID,
        COUNT(*) AS Patient_Count,
        AVG(h.charges) AS Average_Charges,
        AVG(m.BMI) AS Average_BMI,
        AVG(m.HBA1C) AS Average_HBA1C,
        SUM(CASE
            WHEN m.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Heart_Issues_Percentage,
        SUM(CASE
            WHEN m.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Transplants_Percentage,
        SUM(CASE
            WHEN m.Cancer_history = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Cancer_History_Percentage,
        SUM(CASE
            WHEN m.smoker = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Smoker_Percentage
    FROM
        Hospitalisation_details h
            JOIN
        Medical_Examinations m ON h.Customer_ID = m.Customer_ID
    GROUP BY h.State_ID;

-- Aggregated Metrics by State and City Tier
CREATE OR REPLACE VIEW Aggregated_By_State_CityTier AS
    SELECT 
        h.State_ID,
        h.City_tier,
        COUNT(*) AS Patient_Count,
        AVG(h.charges) AS Average_Charges,
        AVG(m.BMI) AS Average_BMI,
        AVG(m.HBA1C) AS Average_HBA1C,
        SUM(CASE
            WHEN m.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Heart_Issues_Percentage,
        SUM(CASE
            WHEN m.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Transplants_Percentage,
        SUM(CASE
            WHEN m.Cancer_history = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Cancer_History_Percentage,
        SUM(CASE
            WHEN m.smoker = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Smoker_Percentage
    FROM
        Hospitalisation_details h
            JOIN
        Medical_Examinations m ON h.Customer_ID = m.Customer_ID
    GROUP BY h.State_ID , h.City_tier;

-- Aggregated Metrics by City Tier
CREATE OR REPLACE VIEW Aggregated_By_CityTier AS
    SELECT 
        h.City_tier,
        COUNT(*) AS Patient_Count,
        AVG(h.charges) AS Average_Charges,
        AVG(m.BMI) AS Average_BMI,
        AVG(m.HBA1C) AS Average_HBA1C,
        SUM(CASE
            WHEN m.Heart_Issues = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Heart_Issues_Percentage,
        SUM(CASE
            WHEN m.Any_Transplants = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Transplants_Percentage,
        SUM(CASE
            WHEN m.Cancer_history = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Cancer_History_Percentage,
        SUM(CASE
            WHEN m.smoker = 'yes' THEN 1
            ELSE 0
        END) / COUNT(*) * 100 AS Smoker_Percentage
    FROM
        Hospitalisation_details h
            JOIN
        Medical_Examinations m ON h.Customer_ID = m.Customer_ID
    GROUP BY h.City_tier;