import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Function to evaluate a regression model and plot predictions vs observations with lat/lon
def evaluate_and_plot_model_with_lat_lon(model, model_name, X_test_scaled, y_test, feature_scaler, lat_column, lon_column):
    """
    Function to evaluate a regression model, plot predictions vs observations with lat/lon,
    save the plot and DataFrame, and return the R² and MAE scores.
    
    Parameters:
    model: Trained regression model.
    model_name (str): Name of the model for saving files.
    X_test_scaled (pd.DataFrame): Scaled test features.
    y_test (pd.DataFrame or pd.Series): True target values for the test set.
    feature_scaler (MinMaxScaler): Scaler used to transform the features.
    lat_column (str): Name of the latitude column in the test set.
    lon_column (str): Name of the longitude column in the test set.
    
    Returns:
    output (pd.DataFrame): DataFrame containing predictions and original features.
    test_r2 (float): R² score for the test set.
    test_mae (float): MAE score for the test set.
    """
    # Generate predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    test_r2 = r2_score(y_true=y_test, y_pred=y_pred)
    test_mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    # Create DataFrame with original (inverse transformed) features, observed, and predicted values
    X_test_original = pd.DataFrame(feature_scaler.inverse_transform(X_test_scaled), columns=X_test_scaled.columns)
    output = X_test_original.copy()
    output['isc_og_obs'] = y_test.values  # Add observed target values
    output['isc_og_pred'] = y_pred  # Add predicted target values
    
    # Create the folder for saving model evaluation results if it doesn't exist
    os.makedirs('model_evaluation', exist_ok=True)
    
    # Save the DataFrame as a CSV file with the model name
    output_file_path = f'model_evaluation/{model_name}_performance.csv'
    output.to_csv(output_file_path, index=False)
    
    # Extract latitude and longitude from the DataFrame
    lat = X_test_original[lat_column]
    lon = X_test_original[lon_column]

    # Determine min and max values for the plot
    max_val = max(y_test.max().values[0], y_pred.max())
    min_val = min(y_test.min().values[0], y_pred.min())

    # Plot predictions vs observations with lat as color and lon as size
    plt.figure(figsize=(14, 10))  # Increased plot size
    sizes = (lon - lon.min()) / (lon.max() - lon.min()) * 100 + 10  # Scale size of dots by lon
    scatter = plt.scatter(y_pred, y_test, c=lat, s=sizes, cmap='viridis', alpha=0.7)  # Plot scatter with scaled size

    # Plot identity line (1:1 line)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2,
             label=f'R² = {np.round(test_r2, 2)}\nMAE = {np.round(test_mae, 2)}')

    # Add a colorbar for latitude
    cbar = plt.colorbar(scatter)
    cbar.set_label('Latitude', fontsize=14)
    
    # Set labels and title
    plt.xlabel('Predicted overgeneration (in %)', fontsize=14)
    plt.ylabel('Observated overgeneration (in %)', fontsize=14)
    plt.title(f'{model_name} Predictions vs Observations', fontsize=18)

    # Add a custom legend for longitude (size of dots)
    for size_value in [lon.min(), (lon.min() + lon.max()) / 2, lon.max()]:
        plt.scatter([], [], c='k', alpha=0.6, s=(size_value - lon.min()) / (lon.max() - lon.min()) * 100 + 10,
                    label=f'Longitude = {size_value:.2f}')
    
    # Show the legend for longitude (size of dots)
    plt.legend(fontsize=12, title="Dot Size (Longitude)")

    # Save the plot in the 'model_evaluation' folder with the model name
    plot_file_path = f'model_evaluation/{model_name}_predictions_vs_observations.png'
    plt.savefig(plot_file_path)

    return output, test_r2, test_mae

# Function to generate and plot feature importance for a given model
def plot_feature_importance(model, model_name, feature_names):
    """
    Function to generate and plot feature importance for a given model.
    Saves the plot and returns the feature importances in a DataFrame.
    
    Parameters:
    model: Trained model that has feature importances or coefficients.
    model_name (str): Name of the model for saving files.
    feature_names (list): List of feature names in the dataset.
    
    Returns:
    feature_importance_df (pd.DataFrame): DataFrame containing features and their importances.
    """
    # Check if the model has feature importance (e.g., tree-based models like Random Forest, LightGBM)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # For linear models like SVR
        importances = np.abs(model.coef_)
    else:
        raise ValueError(f"Model {model_name} does not support feature importances or coefficients.")
    
    # Create a DataFrame for the feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Create the folder for saving model evaluation results if it doesn't exist
    os.makedirs('model_evaluation', exist_ok=True)
    
    # Plotting feature importance
    plt.figure(figsize=(14, 10))  # Set a large figure size
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title(f'{model_name} Feature Importance', fontsize=18)

    # Save the plot in the 'model_evaluation' folder with the model name
    plot_file_path = f'model_evaluation/{model_name}_feature_importance.png'
    plt.savefig(plot_file_path)

    return feature_importance_df
