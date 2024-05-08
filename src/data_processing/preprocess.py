import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import config
import joblib
import torch


def preprocess_data(raw_file_path):
    cols = ['datetime', 'lotusSigh_mt', 'datetime_local', 'lotusMaxBWH_ft']
    numeric_cols = ['lotusSigh_mt', 'lotusMaxBWH_ft']
    df = pd.read_csv(raw_file_path, usecols=cols, parse_dates=['datetime_local'])
    
    df['lotusMaxBWH_ft'] = df['lotusMaxBWH_ft'] * 0.3048 # convert to meters

    # split data
    train = df.iloc[:29800]
    test = df.iloc[29800:]

    # scaling
    numeric_cols = ['lotusSigh_mt', 'lotusMaxBWH_ft']
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    scaled_train = scaler.transform(train[numeric_cols])
    scaled_test = scaler.transform(test[numeric_cols])
    joblib.dump(scaler, './data_processing/scaler.gz') # save the scaler object and y_test unscaled for later figure use in testing 

    x_train, y_train = prep_time_series(scaled_train)
    x_test, y_test = prep_time_series(scaled_test)
    _, y_test_unscaled = prep_time_series(test[numeric_cols])

  
    '''
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    '''

    # y_test_unscaled = scaler.inverse_transform(y_test)
    joblib.dump(y_test_unscaled, './data_processing/y_test_unscaled.gz')

    return torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test) 


def prep_time_series(scaled_data): 
    '''
    one sample will have 
    - input shape: (1, seq_lag, d)
    - ouput shape: (1, seq_next, d)
    where d = dimension of data (# cols in data)

    given N samples, for the entire dataset
    - input shape: (K, seq_lag, d)
    - output shape: (K, seq_next, d)
    where K = N - (seq_lag + seq_delay + seq_next)

    TODO currently programmed for d=2 (ie. one x and one y feature)
    will modify later for potentially multivariable x and y
    '''

    seq_lag = config["sequence_lag"]
    seq_delay = config["sequence_delay"]
    seq_next = config["sequence_next"]

    N = scaled_data.shape[0]
    K = N - (seq_lag + seq_delay + seq_next)

    x_raw = scaled_data[:, 0]
    y_raw = scaled_data[:, 1:]

    x_slice = np.array([range(i, i + seq_lag) for i in range(K)])
    y_slice = np.array([range(i + seq_lag + seq_delay, i + seq_lag + seq_delay + seq_next) for i in range(K)])

    x_sample = x_raw[x_slice]
    y_sample = y_raw[y_slice]

    return x_sample, y_sample
