import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('src/dataset.csv', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_loguniform('C', 1e-5, 1e2)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    # Train model with suggested hyperparameters
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Run Optuna study to find the best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best parameters
print(f'Best parameters: {study.best_params}')

# Train the model with the best parameters
best_model = LogisticRegression(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Save the best model
joblib.dump(best_model, 'models/models.joblib')
