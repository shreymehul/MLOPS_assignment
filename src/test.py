import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import pytest

# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)  # Read the CSV file into a DataFrame
    X = df.iloc[:, :-1]  # Select all columns except the last one as features
    y = df.iloc[:, -1]  # Select the last column as the target variable
    return X, y

# Main function to load data, load the model, and evaluate the model
def main():
    X, y = load_data('data/dataset.csv')  # Load the data from the CSV file
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets
    
    # Load the saved model
    model = joblib.load('models/model.joblib')  # Load the trained model from the file
    
    # Evaluate the model
    y_pred = model.predict(X_test)  # Predict the target variable for the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
    print(f'Accuracy: {accuracy}')  # Print the accuracy

# Entry point of the script
if _name_ == "_main_":
    main()  # Call the main function