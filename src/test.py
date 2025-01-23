import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import pytest

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def main():
    X, y = load_data('data/dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the saved model
    model = joblib.load('models/model.joblib')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check if the accuracy is above a certain threshold
    assert accuracy > 0.7, f"Expected accuracy > 0.7 but got {accuracy}"

if __name__ == "__main__":
    pytest.main()