from preprocessing import preprocess
from train_test_split import get_train_test_split
from train_model import train_random_forest, save_model
from evaluate import evaluate_model


def main():
    print("Loading and preprocessing data...")
    df = preprocess('data/raw/telco_churn.csv')

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)

    print("\nEvaluation on test set:")
    evaluate_model(model, X_test, y_test)

    print("\nSaving model...")
    save_model(model, 'models/churn_rf_model.pkl')
    print("Model saved to models/churn_rf_model.pkl")


if __name__ == "__main__":
    main()