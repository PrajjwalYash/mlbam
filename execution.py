import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

from ML_data_prep import *
from model_selection import train_random_forest_regressor_with_grid_search, train_lgbm_regressor_with_grid_search, train_knn_regressor_with_grid_search, train_svr_with_grid_search
from model_evaluation import *
from end_to_end_isc_pred import ts_prediction_isc_with_mape_and_plot

training_data = pd.read_csv('training_data.csv')
training_data.drop(columns='Unnamed: 0', inplace=True)

# Define lag columns and target variables
lag_cols = ['lat', 'lon']
lags = range(1, 5)
target_variables = ['isc_og_per']

# Create lag features
training_data = create_lag_features(df=training_data, lags=lags, lag_cols=lag_cols)

# Create train/test splits and scale the data
X_train_scaled, y_train, X_test_scaled, y_test, feature_scaler = create_datasplits(training_data, target_variables, split=0.2)


# RF training and evaluation

rf_model = train_random_forest_regressor_with_grid_search(X_train = X_train_scaled, y_train = y_train)
output,test_r2,test_mae = evaluate_and_plot_model_with_lat_lon(model=rf_model, model_name= 'Random forest',X_test_scaled=X_test_scaled, feature_scaler=feature_scaler, lat_column='lat', lon_column='lon', y_test = y_test)
# rf_feature_importance = plot_feature_importance(model=rf_model, model_name = 'Random forest', feature_names=X_test_scaled.columns)
# # LightGBM training and evaluation

# lgb_model = train_lgbm_regressor_with_grid_search(X_train = X_train_scaled, y_train = y_train)
# output,test_r2,test_mae = evaluate_and_plot_model_with_lat_lon(model=lgb_model, model_name= 'LightGBM', X_test_scaled=X_test_scaled, feature_scaler=feature_scaler, lat_column='lat', lon_column='lon', y_test = y_test)
# lgb_feature_importance = plot_feature_importance(model=lgb_model, model_name = 'LightGBM', feature_names=X_test_scaled.columns)

# # SVM training and evaluation

# svm_model = train_svr_with_grid_search(X_train = X_train_scaled, y_train=y_train)
# output,test_r2,test_mae = evaluate_and_plot_model_with_lat_lon(model=svm_model, model_name= 'Support Vector', X_test_scaled=X_test_scaled, feature_scaler=feature_scaler, lat_column='lat', lon_column='lon', y_test = y_test)


# # kNN training and evaluation

# knn_model = train_knn_regressor_with_grid_search(X_train=X_train_scaled,y_train=y_train)
# output,test_r2,test_mae = evaluate_and_plot_model_with_lat_lon(model=knn_model, model_name= 'k- Nearest Neighbour',X_test_scaled=X_test_scaled, feature_scaler=feature_scaler, lat_column='lat', lon_column='lon', y_test = y_test)

#End-to-end prediction of Isc with MLBAM + SPOOPA
test_path = "test_data"
def process_all_files_in_folder(model, feature_scaler, lag, lag_cols, X_test_scaled):
    # Get the list of all files in the folder
    files = os.listdir(test_path)
    
    # Filter for CSV files only
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Process each CSV file
    for file in csv_files:
        if file[:3]=='sc1':
            print(f"Processing file: {file}")
            
            # Call your prediction function with the current file
            try:
                df1, df_mlbam = ts_prediction_isc_with_mape_and_plot(
                    filename=file, 
                    model=model, 
                    feature_scaler=feature_scaler, 
                    lag=lag, 
                    lag_cols=lag_cols, 
                    X_test_scaled=X_test_scaled
                )
                
                # Output results or save them if needed
                print(f"Completed processing for file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
process_all_files_in_folder(
    model=rf_model, 
    feature_scaler=feature_scaler, 
    lag=4, 
    lag_cols=lag_cols, 
    X_test_scaled=X_test_scaled
)
# df1, df_mlbam = ts_prediction_isc_with_mape_and_plot(filename = 'sc1_8jul24_isc', model = rf_model, feature_scaler=feature_scaler, lag = 4, lag_cols = lag_cols, X_test_scaled=X_test_scaled)

