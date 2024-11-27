-- Patient Segmentation

SELECT 
    hd.Customer_ID,
    TIMESTAMPDIFF(YEAR, hd.date, CURDATE()) AS Age,
    hd.children,
    hd.charges,
    hd.Hospital_tier,
    hd.City_tier,
    hd.State_ID,
    me.BMI,
    me.HBA1C,
    me.Heart_Issues,
    me.Any_Transplants,
    me.Cancer_history,
    me.NumberOfMajorSurgeries,
    me.smoker
FROM
    Hospitalisation_details hd
        JOIN
    Medical_Examinations me ON hd.Customer_ID = me.Customer_ID;

-- Create a View for Patient Segmentation
CREATE OR REPLACE VIEW Patient_Segmentation_View AS
    SELECT 
        hd.Customer_ID,
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
        END AS Cancer_history,
        me.NumberOfMajorSurgeries,
        CASE
            WHEN me.smoker = 'yes' THEN 1
            ELSE 0
        END AS smoker
    FROM
        Hospitalisation_details hd
            JOIN
        Medical_Examinations me ON hd.Customer_ID = me.Customer_ID;
