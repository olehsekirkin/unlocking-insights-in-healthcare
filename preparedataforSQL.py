import pandas as pd
import os

# Define file paths
hospitalisation_file = 'Hospitalisation_details.csv'
medical_examinations_file = 'Medical_Examinations.csv'
names_file = 'Names.xlsx'


# 1. Process Hospitalisation_details.csv
def process_hospitalisation_details(input_file, output_file):
    # Read the CSV
    df = pd.read_csv(input_file)

    # Delete rows where 'month' is '?'
    df = df[df['month'] != '?']

    # Delete rows where 'year' is '?'
    df = df[df['year'] != '?']

    # Identify rows with invalid month entries
    valid_months = pd.to_datetime(df['month'], format='%b', errors='coerce')
    invalid_months = df['month'][valid_months.isna()]

    if not invalid_months.empty:
        print("Warning: Found invalid month entries:")
        print(df.loc[valid_months.isna(), ['Customer_ID', 'month', 'year', 'date']])
        # Drop rows with invalid months
        df = df.drop(valid_months[valid_months.isna()].index)
    else:
        print("All month entries are valid.")

    # Convert 'month' to month number with error handling
    df['month_num'] = pd.to_datetime(df['month'], format='%b', errors='coerce').dt.month

    # After conversion, check for any NaT values (should be none after dropping invalid months)
    if df['month_num'].isna().any():
        print("Warning: Some month entries could not be converted and will be dropped.")
        df = df.dropna(subset=['month_num'])

    # Convert 'year' to numeric, coercing errors to NaN
    df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    invalid_years = df['year'][df['year_num'].isna()]

    if not invalid_years.empty:
        print("Warning: Found invalid year entries:")
        print(df.loc[df['year_num'].isna(), ['Customer_ID', 'month', 'year', 'date']])
        # Drop rows with invalid years
        df = df.dropna(subset=['year_num'])
    else:
        print("All year entries are valid.")

    # Create 'day' as numeric, coercing errors to NaN
    df['day_num'] = pd.to_numeric(df['date'], errors='coerce')
    invalid_days = df['day_num'][df['day_num'].isna()]

    if not invalid_days.empty:
        print("Warning: Found invalid day entries:")
        print(df.loc[df['day_num'].isna(), ['Customer_ID', 'month', 'year', 'date']])
        # Drop rows with invalid days
        df = df.dropna(subset=['day_num'])
    else:
        print("All day entries are valid.")

    # Create 'date' in mm/dd/yyyy format
    df['date'] = df.apply(
        lambda row: f"{int(row['month_num']):02d}/{int(row['day_num']):02d}/{int(row['year_num'])}", axis=1
    )

    # Remove unnecessary columns
    df = df.drop(['year', 'month', 'month_num', 'day_num', 'year_num'], axis=1)

    # Modify 'Customer_ID' to remove 'Id' prefix and convert to integer
    df['Customer_ID'] = df['Customer_ID'].str.extract(r'Id(\d+)')
    df = df.dropna(subset=['Customer_ID'])
    df['Customer_ID'] = df['Customer_ID'].astype(int)

    # Modify 'Hospital_tier' and 'City_tier' to extract numbers
    df['Hospital_tier'] = df['Hospital_tier'].str.extract(r'(\d+)')
    df = df.dropna(subset=['Hospital_tier'])
    df['Hospital_tier'] = df['Hospital_tier'].astype(int)

    df['City_tier'] = df['City_tier'].str.extract(r'(\d+)')
    df = df.dropna(subset=['City_tier'])
    df['City_tier'] = df['City_tier'].astype(int)

    # Reorder columns as per desired output
    desired_columns = ['Customer_ID', 'date', 'children', 'charges', 'Hospital_tier', 'City_tier', 'State_ID']
    df = df[desired_columns]

    # Save to new CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} and saved to {output_file}")


# 2. Process Medical_Examinations.csv
def process_medical_examinations(input_file, output_file):
    # Read the CSV
    df = pd.read_csv(input_file)

    # Modify 'Customer_ID' to remove 'Id' prefix and convert to integer
    df['Customer_ID'] = df['Customer_ID'].str.extract(r'Id(\d+)')
    df = df.dropna(subset=['Customer_ID'])
    df['Customer_ID'] = df['Customer_ID'].astype(int)

    # Modify 'NumberOfMajorSurgeries'
    def parse_surgeries(x):
        if isinstance(x, str):
            if 'No major surgery' in x:
                return 0
            else:
                # Extract numbers from the string
                numbers = ''.join(filter(str.isdigit, x))
                return int(numbers) if numbers else 0
        elif pd.isna(x):
            return 0
        else:
            return int(x)

    df['NumberOfMajorSurgeries'] = df['NumberOfMajorSurgeries'].apply(parse_surgeries)

    # Save to new CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} and saved to {output_file}")


# 3. Process Names.xlsx
def process_names(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file)

    # Modify 'Customer_ID' to remove 'Id' prefix and convert to integer
    df['Customer_ID'] = df['Customer_ID'].str.extract(r'Id(\d+)')
    df = df.dropna(subset=['Customer_ID'])
    df['Customer_ID'] = df['Customer_ID'].astype(int)

    # Optionally, you can save it as CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} and saved to {output_file}")


# Define output file names
hospitalisation_output = 'Hospitalisation_details_modified.csv'
medical_examinations_output = 'Medical_Examinatios_modified.csv'
names_output = 'Names_modified.csv'

# Call the processing functions
process_hospitalisation_details(hospitalisation_file, hospitalisation_output)
process_medical_examinations(medical_examinations_file, medical_examinations_output)
process_names(names_file, names_output)
