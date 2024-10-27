import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(red_wine_url, sep=";")

# Separate features and target
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow experiment
mlflow.set_experiment("Wine Quality Experiment")

# Define a list of hyperparameters for experimentation
regularization_strengths = [0.1, 1.0, 10.0]

for C in regularization_strengths:
    with mlflow.start_run():
        # Model training with different hyperparameters
        model = LogisticRegression(C=C, max_iter=200)
        model.fit(X_train, y_train)
        
        # Prediction and accuracy calculation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged model with C={C} and accuracy={accuracy:.4f}")
