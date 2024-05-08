import matplotlib.pyplot as plt
import numpy as np
import joblib
from config import config
import torch
import wandb


def test_lstm(model, x_test_tensor, y_test_tensor):
    model.eval()

    criterion = torch.nn.MSELoss(reduction=config["MSELoss_criterion"])

    with torch.no_grad():
        test_predictions = model(x_test_tensor)

    # reverse scale the prediction values
    scaler = joblib.load('./data_processing/scaler.gz') 





    y_test_unscaled = joblib.load('./data_processing/y_test_unscaled.gz')

    agg_test_preds = average_overlapping_series(test_predictions)
    agg_test_actuals = average_overlapping_series(y_test_unscaled)


 

    print(y_test_unscaled.shape)
    print(type(y_test_unscaled))



def average_overlapping_series(matrix):
    '''
    have an nxm matrix (n rows, m columns)
    each row represents a time series, and rows overlap in time but are different by one step in time
    want an average at each time step

    ex. for n=5, m=4

    t0 t1 t2 t3
    t1 t2 t3 t4
    t2 t3 t4 t5
    t3 t4 t5 t6
    t4 t5 t6 t7

    ex. output [t0_avg, t1_avg, ..., t8_avg]

    '''

    n, m = matrix.shape
    max_time = m + n - 1

    # init list to store the values for each time step
    time_steps = [[] for _ in range(max_time)]

    for i in range(n):
        for j in range(m):
            time_index = i + j
            time_steps[time_index].append(matrix[i, j])

    # get the average for each time step
    averages = [np.mean(values) for values in time_steps]

    return averages 
