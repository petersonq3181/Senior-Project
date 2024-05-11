import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import config
import joblib
import torch


def preprocess_data(raw_file_path):
    cols = ['datetime', 'lotusSigh_mt', 'datetime_local', 'lotusMaxBWH_ft', 'lotusMinBWH_ft', 'tide_ft']
    numeric_cols = ['lotusSigh_mt', 'tide_ft', 'lotusMaxBWH_ft']
    df = pd.read_csv(raw_file_path, usecols=cols, parse_dates=['datetime_local'])
    
    df['lotusMaxBWH_ft'] = df['lotusMaxBWH_ft'] * 0.3048 # convert to meters
    df['tide_ft'] = df['tide_ft'] * 0.3048 # convert to meters
    
    # split data
    train = df.iloc[:29800]
    test = df.iloc[29800:]

    # scaling
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    scaled_train = scaler.transform(train[numeric_cols])
    scaled_test = scaler.transform(test[numeric_cols])
    joblib.dump(scaler, './data_processing/scaler.gz') # save the scaler object and y_test unscaled for later figure use in testing 

    x_train, y_train = prep_time_series(scaled_train)
    x_test, y_test = prep_time_series(scaled_test)
    _, y_test_unscaled = prep_time_series(test[numeric_cols].to_numpy())
    joblib.dump(torch.tensor(y_test_unscaled).squeeze(-1).float(), './data_processing/y_test_unscaled.gz')

    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    x_test_tensor = torch.tensor(x_test).float()
    y_test_tensor = torch.tensor(y_test).float()

    if x_train_tensor.dim() == 23:
        # add a third dimension on the x data (N, sequence_lag) --> (N, sequence_lag, 1)
        x_train_tensor = x_train_tensor.unsqueeze(2)
        x_test_tensor = x_test_tensor.unsqueeze(2)


    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


def prep_time_series(scaled_data): 
    '''
    one sample will have 
    - x shape: (1, seq_lag, input_dim)
    - y shape: (1, seq_next, 1)

    given N samples, for the entire dataset
    - x shape: (K, seq_lag, input_dim)
    - y shape: (K, seq_next, 1)
    where K = N - (seq_lag + seq_delay + seq_next)
    '''

    seq_lag = config["sequence_lag"]
    seq_delay = config["sequence_delay"]
    seq_next = config["sequence_next"]

    N = scaled_data.shape[0]
    K = N - (seq_lag + seq_delay + seq_next)

    x_raw = scaled_data[:, :config['input_dim']]
    y_raw = scaled_data[:, config['input_dim']:]

    x_slice = np.array([range(i, i + seq_lag) for i in range(K)])
    y_slice = np.array([range(i + seq_lag + seq_delay, i + seq_lag + seq_delay + seq_next) for i in range(K)])

    x_sample = x_raw[x_slice]
    y_sample = y_raw[y_slice]

    return x_sample, y_sample
