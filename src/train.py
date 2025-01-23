import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)  # Read the CSV file into a DataFrame
    X = df.iloc[:, :-1]  # Select all columns except the last one as features
    y = df.iloc[:, -1]  # Select the last column as the target variable
    return X, y

# Function to train a logistic regression model
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets
    model = LogisticRegression()  # Initialize the logistic regression model
    model.fit(X_train, y_train)  # Train the model on the training data
    y_pred = model.predict(X_test)  # Predict the target variable for the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
    print(f'Accuracy: {accuracy}')  # Print the accuracy
    return model  # Return the trained model

# Function to save the trained model to a file
def save_model(model, filepath):
    joblib.dump(model, filepath)  # Save the model to the specified file

# Main function to load data, train the model, and save the model
def main():
    X, y = load_data('data/dataset.csv')  # Load the data from the CSV file
    model = train(X, y)  # Train the model
    save_model(model, 'models/model.joblib')  # Save the trained model

# Entry point of the script
if _name_ == "_main_":
    main()  # Call the main function