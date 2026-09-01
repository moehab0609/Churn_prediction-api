from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print a full evaluation report: confusion matrix + precision/recall/F1."""
    predictions = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=['No Churn', 'Churn']))