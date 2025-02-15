import pandas as pd
from logger import logger


def explore_data(df, categorical_columns):
    """
    Explores the dataset and converting categorical columns to object type,
    checking for missing values, and identifying duplicates.

    """
    logger.info("Starting data exploring process...")

    # Convert categorical columns to object type
    df[categorical_columns] = df[categorical_columns].astype('object')
    logger.info("Converted categorical columns to object type.")

    # Descriptive statistics
    logger.info(f"Descriptive Statistics:\n{df.describe()}")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")

    # Check for missing values
    missing_values = df.isna().sum()
    logger.info(f"Missing values per column:\n{missing_values}")

    logger.info("Data exploring process completed.")
    return df

def fill_missing_bmi(df, glucose_col='avg_glucose_level', bmi_col='bmi', gender_col='gender', bins=4):
    """
    Fills missing BMI values with the median BMI, grouped by gender and glucose level bins.
    If any BMI values remain missing, they are filled using the overall median BMI per gender.
    """
    logger.info("Starting BMI filling process...")

    df = df.copy()

    # Create glucose bins
    df['Glucose_bin'] = pd.cut(df[glucose_col], bins=bins, labels=False)
    logger.info(f"Created {bins} glucose bins.")

    # Calculate median BMI for each (gender, glucose_bin) group
    median_bmi = df.groupby([gender_col, 'Glucose_bin'])[bmi_col].median().reset_index()
    median_bmi = median_bmi.rename(columns={bmi_col: 'Median_BMI'})
    logger.info("Calculated median BMI for each (gender, glucose_bin) group.")

    # Merge and fill missing BMI values
    df = pd.merge(df, median_bmi, on=[gender_col, 'Glucose_bin'], how='left')
    missing_before = df[bmi_col].isna().sum()
    df[bmi_col] = df[bmi_col].fillna(df['Median_BMI'])
    missing_after = df[bmi_col].isna().sum()
    logger.info(f"Filled {missing_before - missing_after} missing BMI values.")

    # Handle any remaining NaN values using gender-wise median BMI
    overall_median_per_gender = df.groupby(gender_col)[bmi_col].transform(lambda x: x.fillna(x.median()))
    df[bmi_col] = df[bmi_col].fillna(overall_median_per_gender)
    final_missing = df[bmi_col].isna().sum()
    
    if final_missing > 0:
        overall_median = df[bmi_col].median()
        df[bmi_col].fillna(overall_median, inplace=True)
        logger.info(f"Filled remaining {final_missing} missing BMI values using overall median.")

    # Drop temporary columns
    df.drop(columns=['Glucose_bin', 'Median_BMI'], inplace=True)
    logger.info("Dropped temporary columns used for grouping.")

    logger.info("BMI filling process completed.")
    return df

def encode_columns(df, encoding_dict):
    """
    Encodes categorical columns using a predefined mapping.

    """
    logger.info("Starting encoding process...")

    df_copy = df.copy()
    for column, mapping in encoding_dict.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].map(mapping).astype(int)
            logger.info(f"Encoded column: {column}")

    logger.info("Encoding process completed.")
    return df_copy
