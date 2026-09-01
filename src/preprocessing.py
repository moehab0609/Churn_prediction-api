import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw churn data from CSV."""
    return pd.read_csv(filepath)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Fix TotalCharges: convert to numeric, fill missing (tenure=0 customers) with 0."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def preprocess(filepath: str) -> pd.DataFrame:
    """Full preprocessing pipeline: load, clean, encode, and finalize data."""
    df = load_data(filepath)
    df = clean_total_charges(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = finalize_features(df)
    return df

def encode_binary_columns(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Label-encode binary Yes/No columns and gender. Set include_target=False at inference time, when no label column exists."""
    df = df.copy()
    
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    if include_target:
        binary_cols = binary_cols + ['Churn']
    
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
    
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-category columns."""
    df = df.copy()
    
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    
    return df

def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-predictive columns and ensure consistent numeric dtypes."""
    df = df.copy()
    df = df.drop(columns=['customerID'])
    
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df



def preprocess_single_record(record: dict, feature_columns: list) -> pd.DataFrame:
    """Preprocess a single raw customer record into model-ready features, aligned to training columns."""
    df = pd.DataFrame([record])
    df = clean_total_charges(df)
    df = encode_binary_columns(df, include_target=False)
    df = encode_categorical_columns(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df