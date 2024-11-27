-- 1. Create the Database
CREATE DATABASE IF NOT EXISTS healthcare_data;
USE healthcare_data;

-- 2. Create Tables

CREATE TABLE IF NOT EXISTS Hospitalisation_details (
    Customer_ID INT PRIMARY KEY,
    date DATE,
    children INT,
    charges DECIMAL(10 , 2 ),
    Hospital_tier INT,
    City_tier INT,
    State_ID VARCHAR(10)
);

-- Table: Medical_Examinatios
CREATE TABLE IF NOT EXISTS Medical_Examinations (
    Customer_ID INT PRIMARY KEY,
    BMI DECIMAL(6 , 3 ),
    HBA1C DECIMAL(4 , 2 ),
    Heart_Issues VARCHAR(3),
    Any_Transplants VARCHAR(3),
    Cancer_history VARCHAR(3),
    NumberOfMajorSurgeries INT,
    smoker VARCHAR(10)
);

-- Table: Names
CREATE TABLE IF NOT EXISTS Names (
    Customer_ID INT PRIMARY KEY,
    name VARCHAR(255)
);

-- 3. Load Data from CSV Files

-- Load Hospitalisation_details_modified.csv
LOAD DATA LOCAL INFILE 'C:/Users/olehs/Desktop/Healthcare_Project/Hospitalisation_details_modified.csv'
INTO TABLE Hospitalisation_details
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Customer_ID, @date, children, charges, Hospital_tier, City_tier, State_ID)
SET date = STR_TO_DATE(@date, '%m/%d/%Y');

-- Load Medical_Examinations_modified.csv
LOAD DATA LOCAL INFILE 'C:/Users/olehs/Desktop/Healthcare_Project/Medical_Examinations_modified.csv'
INTO TABLE Medical_Examinations
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Customer_ID, BMI, HBA1C, Heart_Issues, Any_Transplants, Cancer_history, NumberOfMajorSurgeries, smoker);

-- Load Names_modified.csv
LOAD DATA LOCAL INFILE 'C:/Users/olehs/Desktop/Healthcare_Project/Names_modified.csv'
INTO TABLE Names
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(Customer_ID, name);

-- Detecting outliers in BMI using IQR
WITH ordered_bmi AS (
    SELECT BMI, 
           ROW_NUMBER() OVER (ORDER BY BMI) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM Medical_Examinations
),
quartiles AS (
    SELECT 
        MAX(CASE WHEN row_num = FLOOR(0.25 * total_count) THEN BMI END) AS Q1,
        MAX(CASE WHEN row_num = FLOOR(0.75 * total_count) THEN BMI END) AS Q3
    FROM ordered_bmi
),
iqr AS (
    SELECT Q3 - Q1 AS IQR_value
    FROM quartiles
)
SELECT 
    o.BMI
FROM 
    ordered_bmi o
CROSS JOIN quartiles q
CROSS JOIN iqr
WHERE 
    o.BMI < q.Q1 - 1.5 * iqr.IQR_value
    OR 
    o.BMI > q.Q3 + 1.5 * iqr.IQR_value;