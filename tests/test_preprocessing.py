import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest
from preprocessing import clean_total_charges, encode_binary_columns, preprocess_single_record


def test_clean_total_charges_fills_blank_with_zero():
    """A blank TotalCharges (new customer, tenure=0) should become 0.0, not crash or stay blank."""
    df = pd.DataFrame({'TotalCharges': [' ', '29.85', '100.5']})
    result = clean_total_charges(df)

    assert result['TotalCharges'].iloc[0] == 0.0
    assert result['TotalCharges'].dtype == float


def test_clean_total_charges_preserves_valid_values():
    """Valid numeric strings should convert correctly, not just default to 0."""
    df = pd.DataFrame({'TotalCharges': ['150.75']})
    result = clean_total_charges(df)

    assert result['TotalCharges'].iloc[0] == 150.75


def test_encode_binary_columns_without_target():
    """At inference time (include_target=False), missing Churn column should not raise an error."""
    df = pd.DataFrame({
        'gender': ['Male'],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'PhoneService': ['Yes'],
        'PaperlessBilling': ['No'],
    })
    result = encode_binary_columns(df, include_target=False)

    assert result['gender'].iloc[0] == 1
    assert result['Partner'].iloc[0] == 1
    assert result['Dependents'].iloc[0] == 0


def test_preprocess_single_record_matches_training_columns():
    """A single incoming record must be aligned to have exactly the training feature columns, in order."""
    feature_columns = ['tenure', 'MonthlyCharges', 'Contract_One year', 'Contract_Two year']

    record = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 12, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 55.0, 'TotalCharges': '660.0'
    }

    result = preprocess_single_record(record, feature_columns)

    assert list(result.columns) == feature_columns
    assert result['Contract_One year'].iloc[0] == 1
    assert result['Contract_Two year'].iloc[0] == 0