from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a Random Forest baseline with class weighting to address imbalance."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numeric features using StandardScaler, fit only on training data."""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train a baseline logistic regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath: str) -> None:
    """Save a trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load a trained model from disk."""
    return joblib.load(filepath)


def save_feature_columns(columns, filepath: str) -> None:
    """Save the exact training feature column order for later alignment."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(columns), filepath)


def load_feature_columns(filepath: str) -> list:
    """Load the saved training feature column order."""
    return joblib.load(filepath)