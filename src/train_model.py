from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


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