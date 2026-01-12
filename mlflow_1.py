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

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_neighbors, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("max_depth", max_depth)


    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
   
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')


    # save confusion matrix plot
    plt.savefig('confusion_matrix.png')

    # Log artifact
    mlflow.log_artifact('confusion_matrix.png')
  
    # Log additional metrics if needed
    mlflow.log_artifact(__file__)  # Log the current script as an artifact

    # Tag the run
    mlflow.set_tag("model_type", "RandomForestClassifier")

    # Log the model
    mlflow.sklearn.log_model(rf, "random_forest_model")
    print(f"Model accuracy: {accuracy}")



