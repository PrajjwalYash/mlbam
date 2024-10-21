import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

def train_random_forest_regressor_with_grid_search(X_train, y_train):
    """
    Train a Random Forest Regressor with GridSearchCV and save the best model.
    """
    # Define the parameter grid for RandomForestRegressor with fewer grid points
    param_grid = {
        'n_estimators': [100, 300],            # Number of trees
        'max_depth': [30],                 # Maximum depth of trees
        'min_samples_split': [5],          # Minimum samples required to split a node
        'min_samples_leaf': [1],            # Minimum samples required at a leaf node
        'bootstrap': [True],                   # Use bootstrap sampling
        'max_samples': [0.8]              # Row sampling: proportion of samples to draw for training
    }

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf_regressor = grid_search.best_estimator_

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Create folder 'trained_models_with_lag' if it doesn't exist
    os.makedirs('trained_models_with_lag', exist_ok=True)

    # Save the best model to the 'trained_models_with_lag' folder using pickle
    model_path = os.path.join('trained_models_with_lag', 'rf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_rf_regressor, f)

    print(f"Model saved at: {model_path}")

    return best_rf_regressor


def train_lgbm_regressor_with_grid_search(X_train, y_train):
    """
    Train a LightGBM Regressor with GridSearchCV and save the best model.
    """
    # Define the parameter grid for LightGBM with regularization and GOSS
    param_grid = {
        'boosting_type': ['goss'],     # Gradient Boosting (traditional) and GOSS
        'n_estimators': [200],            # Number of boosting iterations
        'max_depth': [30, 50],                 # Maximum depth of trees
        'linear_trees': [False],
        'learning_rate': [0.1],          # Learning rate
        'num_leaves': [50, 70],                # Number of leaves in one tree
        'min_data_in_leaf': [20],          # Minimum data in one leaf
        'lambda_l1': [0.1, 0.3],          # L1 regularization
        'lambda_l2': [0.1],          # L2 regularization
        'min_split_gain': [0.1],          # Minimum gain to make a split
        'subsample': [0.8, 1.0],              # Row sampling (ignored for GOSS)
        'verbose': [-1]
    }

    # Initialize the LightGBM Regressor
    lgbm_regressor = lgb.LGBMRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=lgbm_regressor, param_grid=param_grid,
                               cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_lgbm_regressor = grid_search.best_estimator_

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Create folder 'trained_models_with_lag' if it doesn't exist
    os.makedirs('trained_models_with_lag', exist_ok=True)

    # Save the best model to the 'trained_models_with_lag' folder using pickle
    model_path = os.path.join('trained_models_with_lag', 'lgbm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_lgbm_regressor, f)

    print(f"Model saved at: {model_path}")

    return best_lgbm_regressor


def train_svr_with_grid_search(X_train, y_train):
    """
    Train an SVR model with GridSearchCV and save the best model.
    """
    # Define the parameter grid for SVR with regularization and kernel options
    param_grid = {
        'C': [1, 10],                    # Regularization parameter
        'epsilon': [0.01, 0.1, 0.5],     # Epsilon in the epsilon-SVR model
        'kernel': ['rbf'],               # Kernel type (linear or RBF)
        'gamma': ['scale', 'auto']       # Kernel coefficient for RBF
    }

    # Initialize the SVR model
    svr = SVR()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid,
                               cv=4, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svr = grid_search.best_estimator_

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Create folder 'trained_models_with_lag' if it doesn't exist
    os.makedirs('trained_models_with_lag', exist_ok=True)

    # Save the best model to the 'trained_models_with_lag' folder using pickle
    model_path = os.path.join('trained_models_with_lag', 'svr_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_svr, f)

    print(f"Model saved at: {model_path}")

    return best_svr


def train_knn_regressor_with_grid_search(X_train, y_train):
    """
    Train a KNN Regressor with GridSearchCV and save the best model.
    """
    # Define the parameter grid for KNN
    param_grid = {
        'n_neighbors': [5, 10, 15],         # Number of neighbors to use
        'weights': ['uniform', 'distance'], # Weight function used in prediction
        'metric': ['manhattan', 'minkowski'],  # Distance metric
        'p': [2, 3, 5],                        # Power parameter for Minkowski metric (1: Manhattan, 2: Euclidean)
    }

    # Initialize the KNN Regressor
    knn_regressor = KNeighborsRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=knn_regressor, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_knn_regressor = grid_search.best_estimator_

    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Create folder 'trained_models_with_lag' if it doesn't exist
    os.makedirs('trained_models_with_4lag', exist_ok=True)

    # Save the best model to the 'trained_models_with_4lag' folder using pickle
    model_path = os.path.join('trained_models_with_4lag', 'knn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_knn_regressor, f)

    print(f"Model saved at: {model_path}")

    return best_knn_regressor
