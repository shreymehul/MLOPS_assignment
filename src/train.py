import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)

def main():
    X, y = load_data('data/dataset.csv')
    model = train(X, y)
    save_model(model, 'models/model.joblib')

if __name__ == "__main__":
    main()