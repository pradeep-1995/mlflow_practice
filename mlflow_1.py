import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Mlfow start
import mlflow

# Dagshub integration
import dagshub
dagshub.init(repo_owner='pradeep-1995', repo_name='mlflow_practice', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/pradeep-1995/mlflow_practice.mlflow")


# load dataset
data = load_wine()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter for model
n_neighbors = 4
max_depth = 4

# Enable autologging
mlflow.autolog()
mlflow.set_experiment("Wine_Classification_Experiment")

rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_neighbors, random_state=42)
rf.fit(X_train, y_train)

with mlflow.start_run():

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Log additional metrics if needed
    mlflow.log_artifacts(__file__)  # Log the current script as an artifact



