from sklearn.model_selection import train_test_split
import pandas as pd


def split_features_target(df: pd.DataFrame, target_col: str = 'Churn'):
    """Separate features (X) from target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_train_test_split(df: pd.DataFrame, target_col: str = 'Churn', 
                          test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets, stratified on the target to preserve class balance."""
    X, y = split_features_target(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test