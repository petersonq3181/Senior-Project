import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch

def preprocess_data(raw_file_path):
    cols = ['datetime', 'lotusSigh_mt', 'datetime_local', 'lotusMaxBWH_ft', 'lotusMinBWH_ft']
    numeric_cols = ['lotusSigh_mt', 'lotusMaxBWH_ft']
    df = pd.read_csv(raw_file_path, usecols=cols, parse_dates=['datetime_local'])
    
    df['lotusAvgBHW_ft'] = df[['lotusMaxBWH_ft', 'lotusMinBWH_ft']].mean(axis=1)
    df['lotusAvgBHW_ft'] = df['lotusAvgBHW_ft'] * 0.3048 # convert to meters

    # split data
    train = df.iloc[:29800]
    test = df.iloc[29800:]

    # scaling
    numeric_cols = ['lotusSigh_mt', 'lotusAvgBHW_ft']
    scaler = MinMaxScaler()
    scaler.fit(train[numeric_cols])
    scaled_train = scaler.transform(train[numeric_cols])
    scaled_test = scaler.transform(test[numeric_cols])

    # save the scaler object and y_test unscaled for later figure use in testing 
    joblib.dump(scaler, './data_processing/scaler.gz')
    
    x_train, y_train = create_sequences(scaled_train[:, 0].reshape(-1, 1), scaled_train[:, 1], 12)
    x_test, y_test = create_sequences(scaled_test[:, 0].reshape(-1, 1), scaled_test[:, 1], 12)

    y_test_unscaled = scaler.inverse_transform(np.concatenate((np.zeros_like(y_test.reshape(-1, 1)), y_test.reshape(-1, 1)), axis=1))[:, 1]
    joblib.dump(y_test_unscaled, './data_processing/y_test_unscaled.gz')

    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    x_test_tensor = torch.tensor(x_test).float()
    y_test_tensor = torch.tensor(y_test).float()

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

def create_sequences(input_data, target_data, seq_len):
    xs, ys = [], []
    for i in range(len(input_data) - seq_len):
        xs.append(input_data[i:(i + seq_len)])
        ys.append(target_data[i + seq_len])
    return np.array(xs), np.array(ys)
