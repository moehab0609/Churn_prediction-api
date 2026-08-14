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
    """Full preprocessing pipeline: load raw data and apply all cleaning steps."""
    df = load_data(filepath)
    df = clean_total_charges(df)
    return df