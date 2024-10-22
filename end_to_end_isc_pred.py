import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
L_I = 0.989
folder_path = "test_data"

def mean_absolute_percentage_error(y_true, y_pred):
    """Function to calculate Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_lag_features(df, lags, lag_cols):
    for col in lag_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def get_data(cleanfilename, frame_no):
    print('Data preprocessing initiated')
    
    # Construct the full path to the file
    full_path = os.path.join(folder_path, cleanfilename)
    
    df = pd.read_csv(full_path)
    df['Timestamp'] = pd.to_datetime(df.Timestamp, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.set_index('Timestamp')
    
    if any("duty_1" in s for s in df.columns) and any("duty_2" in s for s in df.columns):
        df['duty_1'] = df['duty_1'] / 100
        df['duty_2'] = df['duty_2'] / 100
        df = df.rename(columns={'duty_1': 'str_sts_01', 'duty_2': 'str_sts_02'})
    
    df = df.reindex(sorted(df.columns), axis=1)
    bfill_params = ['sags']
    df_bfill = df[bfill_params]
    df_bfill = df_bfill.fillna(method='bfill')
    df = df.drop(columns=bfill_params)
    df = df.fillna(method='ffill')
    df = pd.merge(df, df_bfill, on='Timestamp')
    df = df.dropna(axis=0)
    
    df1 = df[df['frame'] == int(frame_no)]
    df1.dropna(axis=0, how='any', inplace=True)
    df1 = df1.resample('30S').mean()
    df1.dropna(axis=0, how='all', inplace=True)
    
    data_start = (df1.index[0]).strftime('%d-%b-%yT%H_%M')
    data_end = (df1.index[-1]).strftime('%d-%b-%yT%H_%M')
    print('Data preprocessing completed')
    
    return df1, data_start, data_end

def get_all_points(df1):
    print('Getting SAA, eclipse and PL operation')
    if any("pnl11_pr_temp" in s for s in df1.columns) & any("pnl22_pr_temp" in s for s in df1.columns):
        df1= df1.rename(columns = {'pnl11_pr_temp':'prt_11', 'pnl22_pr_temp':'prt_22'})
    df1['sun_ang'] = df1['spss_1']
    df1['ecl'] = df1['sags']<50
    df1['pre_ecl'] = True
    df1['post_ecl'] = True
    for i in range(len(df1)-8):
        df1['pre_ecl'][i] = df1['ecl'][i+8]==True
    for i in range(len(df1)-8):
        df1['post_ecl'][i+8] = df1['ecl'][i]==True
    df1['pre_ecl'] = (df1['pre_ecl']==True) & (df1['ecl']== False)
    df1['post_ecl'] = (df1['post_ecl']==True) & (df1['ecl']== False)
    df1['sun_lit'] = (df1['ecl']==False) & (df1['pre_ecl']== False) & (df1['post_ecl']== False)
    df1['PL'] = df1['sun_ang']>60
    print('Success')
    return df1

def z_transform(theta) -> np.array:
    # theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def get_lat_lon(df1):
    print('Getting lat and lon')
    x = df1['sc_x_pos'] # example x-coordinate in km
    y = df1['sc_y_pos'] # example y-coordinate in km
    z = df1['sc_z_pos'] # example z-coordinate in km

    sidAng = 3.158594230000000*180/np.pi # rad
    sidEpoch = pd.Timestamp("2024-3-23 0:0:0.0")
    siderial_rate = 1.140792231481478e-05 + 360/86400 # deg/s

    df1['Sidereal (rad)'] = np.deg2rad( (sidAng  + siderial_rate * ((df1.index - sidEpoch)/np.timedelta64(1, 's'))) % 360 )

    for i in range(0, len(df1)):
        sat_pos_vec = np.array([    x[i],
                                    y[i],
                                    z[i]])

        sat_pos_vec = np.reshape(sat_pos_vec, (3, -1))
        sat_pos_vec = np.dot(z_transform(df1['Sidereal (rad)'][i]), sat_pos_vec)        # in the ECEF frame
        x[i] = sat_pos_vec[0]
        y[i] = sat_pos_vec[1]
        z[i] = sat_pos_vec[2]
    longitude = np.arctan2(y, x)
    altitude = np.sqrt(x*x + y*y + z*z)
    latitude = np.arcsin(z/altitude)

    df1['lon'] = (longitude)*180/np.pi
    df1['lat'] = (latitude)*180/np.pi
    df1['alt'] = (altitude) - 6371
    print('Success')
    return df1

def cell_temp_voc(df1):
    from itertools import chain
    print('Getting temperature')
    inn_str_no = 14
    out_str_no = 14
    temp = ([[((df1['prt_22'])).values - df1['prt_11'].values + df1['cell_temp']]*inn_str_no, [df1['cell_temp'].values]*out_str_no])
    temp = list(chain(*temp))
    temp = np.array(temp).T
    ref = 28
    print('Success')
    return temp, ref

L_I = 0.989

def exp_isc(df1, temp, ref, L_I=L_I):
    print('Isc time-series plot')
    df1['exp_isc'] = 0.5122416
    df1['exp_isc'] = (df1['exp_isc']+(temp[:,-1]-ref)*0.000358)
    doy = df1.index.strftime('%j').astype('int')[0]
    sun_int_factor = folder_path + r"/sun_int_fac.csv"
    df_int = pd.read_csv(sun_int_factor)
    df_int.dropna(how = 'all', axis = 0, inplace = True)
    df_int.dropna(how = 'all', axis = 1, inplace = True)
    int_fac = df_int['intensity factor'][df_int['doy']==doy].values
    df1['doy'] = doy
    df1['month'] = df1.index.strftime('%B')
    df1['exp_isc'] = L_I*int_fac*df1['exp_isc']
    df1['exp_isc'] = df1['exp_isc']*np.cos(df1['sun_ang']*np.pi/180)
    df1['exp_isc'][df1['ecl']==True] = min(df1['isc'])
    df1['isc_og_per'] = ((df1['isc']/df1['exp_isc'])-1)*100
    df1.reset_index(inplace = True)
    return df1


def ts_prediction_isc_with_mape_and_plot(filename, model, feature_scaler, lag, lag_cols, X_test_scaled):
    """
    Function to perform time series prediction of Isc using a given ML model, 
    calculate MAPE, and plot the results. The plot title will include the start 
    and end date of the data, and MAPE values will be shown in the legend.

    Parameters:
    - filename: The base filename to load the data.
    - model: The trained machine learning model to use for predictions.
    - feature_scaler: Scaler used to scale the feature columns.
    - lag: The number of lag features to create.
    - lag_cols: The columns to use for lag feature generation.
    - X_test_scaled: The scaled version of the test dataset (used to get feature columns).

    Returns:
    - df1: The original dataframe with predictions.
    - df_mlbam: The dataframe with scaled features and predictions.
    """
    
    # Load and process the data
    df1, data_start, data_end = get_data(filename, frame_no=29)
    df1 = get_all_points(df1)
    df1 = get_lat_lon(df1)
    temp, ref = cell_temp_voc(df1)
    df1 = exp_isc(df1, temp, ref, L_I=L_I)
    
    # Filter for sunlit and non-PL points
    df1 = df1[(df1['sun_lit'] == True) & (df1['PL'] == False)]
    
    # Create lag features
    lags = range(1, lag + 1)
    df1 = create_lag_features(df=df1, lags=lags, lag_cols=lag_cols)
    df1.dropna(inplace=True)
    
    # Extract feature columns based on X_test_scaled columns
    feature_cols = X_test_scaled.columns
    df_mlbam = df1[feature_cols]
    
    # Scale the feature columns using the provided scaler
    test_data = feature_scaler.transform(df_mlbam)
    
    # Make predictions using the trained model
    pred = model.predict(test_data)
    df_mlbam['isc_og_pred'] = pred
    
    # Add predictions back to the original dataframe
    df1['isc_og_pred'] = df_mlbam['isc_og_pred']
    
    # Calculate predicted Isc
    df1['pred_isc'] = df1['exp_isc'] * (1 + (df1['isc_og_pred'] / 100))
    
    # Calculate MAPE for Isc vs exp_isc and Isc vs pred_isc
    mape_exp_isc = np.round(mean_absolute_percentage_error(df1['isc'], df1['exp_isc']),2)
    mape_pred_isc = np.round(mean_absolute_percentage_error(df1['isc'], df1['pred_isc']),2)
    
    # Extract start and end date from the first and last timestamps
    start_date = df1['Timestamp'].iloc[0].strftime('%d-%b-%Y')
    end_date = df1['Timestamp'].iloc[-1].strftime('%d-%b-%Y')
    
    # Plot the observed, SPOOPA predicted, and MLBAM+SPOOPA predicted Isc
    plt.figure(figsize=(18, 12))  # Increase plot size for better visibility
    plt.plot(df1['Timestamp'][-200:], df1['isc'][-200:], marker='*', label=f'Observed Isc')
    plt.plot(df1['Timestamp'][-200:], df1['exp_isc'][-200:], marker='*', 
             label=f'SPOOPA predicted Isc (MAPE: {mape_exp_isc}%)')
    plt.plot(df1['Timestamp'][-200:], df1['pred_isc'][-200:], marker='*', 
             label=f'MLBAM+SPOOPA predicted Isc (MAPE: {mape_pred_isc}%)')
    
    # Set plot labels and title
    plt.xlabel('Timestamp', fontsize=20)
    plt.ylabel('Isc', fontsize=20)
    plt.title(f'MLBAM improved Isc prediction from {start_date} to {end_date}', fontsize=20)
    
    # Show legend and grid
    plt.legend(fontsize=16)
    plt.grid(True)
    
    # Save the plot in the 'model_evaluation' folder
    os.makedirs('model_evaluation', exist_ok=True)
    plot_file_path = f'model_evaluation/{filename}_MLBAM_Isc_prediction_{start_date}_to_{end_date}.png'
    plt.savefig(plot_file_path)
    
    # Display the plot
    # plt.show()
    
    return df1, df_mlbam