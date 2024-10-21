import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

# Function to create lag features
def create_lag_features(df, lags, lag_cols):
    """
    Adds lagged features to the dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    lags (range): Range of lags to apply.
    lag_cols (list): List of columns to create lags for.
    
    Returns:
    pd.DataFrame: Dataframe with lagged features.
    """
    for col in lag_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

# Function to create data splits and scale features
def create_datasplits(training_data, target_variables, split):
    """
    Splits the dataset into training and testing sets, scales the features, and saves the feature scaler.
    
    Parameters:
    training_data (pd.DataFrame): The input dataset.
    target_variables (list): List of target variables (columns to predict).
    split (float): The test set size as a proportion of the dataset (between 0 and 1).
    
    Returns:
    X_train_scaled (pd.DataFrame): Scaled training features.
    y_train (pd.Series or pd.DataFrame): Training targets.
    X_test_scaled (pd.DataFrame): Scaled testing features.
    y_test (pd.Series or pd.DataFrame): Testing targets.
    feature_scaler (MinMaxScaler): The scaler used to normalize features.
    """
    # Remove rows where the absolute value of 'isc_og_per' is greater than 10
    training_data.loc[abs(training_data['isc_og_per']) > 10] = np.nan
    training_data.dropna(inplace=True)
    
    # Split features (X) and targets (y)
    X = training_data.drop(target_variables, axis=1)
    y = training_data[target_variables]
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=True, random_state=42)
    
    # Scale the features using MinMaxScaler
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = feature_scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the feature scaler to a pickle file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feature_scaler_filename = f'feature_scaler_with_4lags_{timestamp}.pkl'
    with open(feature_scaler_filename, 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    # Return the scaled features and targets
    return X_train_scaled, y_train, X_test_scaled, y_test, feature_scaler

# Example usage:
# Load your training data
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
