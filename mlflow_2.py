from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Mlfow start
import mlflow

# Dagshub integration
import dagshub
dagshub.init(repo_owner='pradeep-1995', repo_name='mlflow_practice', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/pradeep-1995/mlflow_practice.mlflow")

# load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model creation
rf = RandomForestClassifier(random_state=42)

# Grid search parameters
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 4, 6, 8]
}

# Apply grid search with mlflow autologging
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Enable autologging

with mlflow.start_run() as parent_run:
    mlflow.autolog()
    mlflow.set_experiment("Breast_Cancer_Classification_Experiment")

    grid_search.fit(X_train, y_train)

    # log all child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child_run:

            mlflow.log_param("param_grid", grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])


    # Display best parameters and accuracy
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # log params and metrics of the best model
    mlflow.log_params(best_params)

    # Log metrics of the best model
    mlflow.log_metric("best_accuracy", best_score)

    # Log training dataset
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)


    # Log test dataset
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)


    # Log Source Code
    mlflow.log_artifacts(__file__)  # Log the current script as an artifact

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")

    # Tag the run
    mlflow.set_tag("model_type", "RandomForestClassifier_GridSearchCV")

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score}")
    
    