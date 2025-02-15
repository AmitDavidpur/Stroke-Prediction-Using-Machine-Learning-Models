import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from Preprocessing.preprocessing import explore_data, fill_missing_bmi, encode_columns
from logger import logger

@pytest.fixture

def sample_df():
    """ Creates a sample DataFrame for testing """
    data = {
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'age': [45, 30, 50, 29, 60],
        'bmi': [25.0, None, 30.5, 28.0, None],
        'avg_glucose_level': [80.5, 95.2, 70.0, 88.1, 110.3],
        'smoking_status': ['never smoked', 'smokes', 'formerly smoked','smokes', 'smokes']
    }
    return pd.DataFrame(data)

def test_explore_data(sample_df):
    """ Tests the explore_data function """
    categorical_columns = ['gender', 'smoking_status']
    df = explore_data(sample_df.copy(), categorical_columns)

    assert df[categorical_columns].dtypes.eq('object').all(), "Categorical columns should be of type object"
    assert df.duplicated().sum() == 0, "Function should not introduce duplicates"
    assert df.isna().sum().sum() >= 0, "Missing values check should be performed"


def test_fill_missing_bmi(sample_df):
    """Tests the fill_missing_bmi function."""
    df = fill_missing_bmi(sample_df.copy())

    print(df['bmi'])  # Debug print to check if NaN values remain

    assert df['bmi'].isna().sum() == 0, "Function should fill all missing BMI values"
    assert df['bmi'].dtype in [float, int], "BMI values should be numeric"
    assert (df['bmi'] >= 10).all() and (df['bmi'] <= 60).all(), "BMI values should be in a reasonable range"
    
def test_encode_columns():
    """ Tests the encode_columns function """
    df = pd.DataFrame({'gender': ['Male', 'Female', 'Female', 'Male']})
    encoding_dict = {'gender': {'Male': 1, 'Female': 0}}

    df_encoded = encode_columns(df.copy(), encoding_dict)

    assert set(df_encoded['gender'].unique()) == {0, 1}, "Encoded values should be 0 or 1"
    assert df_encoded['gender'].dtype == int, "Encoded column should be of type int"

