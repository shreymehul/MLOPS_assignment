import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)  # Read the CSV file into a DataFrame
    X = df.iloc[:, :-1]  # Select all columns except the last one as features
    y = df.iloc[:, -1]  # Select the last column as the target variable
    return X, y

# Function to save results to a text file
def save_results(accuracy):
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Define the result file path
    result_file = 'results/accuracy3.txt'
    
    # Save the accuracy result to the file
    with open(result_file, 'w') as file:
        file.write(f'Accuracy: {accuracy}\n')

# Main function to load data, load the model, and evaluate the model
def main():
    X, y = load_data('data/dataset.csv')  # Load the data from the CSV file
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and testing sets
    
    # Load the saved model
    model = joblib.load('models/models3.joblib')  # Load the trained model from the file
    
    # Evaluate the model
    y_pred = model.predict(X_test)  # Predict the target variable for the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
    
    # Print accuracy to console
    print(f'Accuracy: {accuracy}')  # Print the accuracy
    
    # Save the accuracy result to a file
    save_results(accuracy)

# Entry point of the script
if __name__ == "__main__":  # Ensure the entry point is "__main__"
    main()  # Call the main function
