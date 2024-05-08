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

        test_loss = criterion(test_predictions.squeeze(-1), y_test_tensor.squeeze(-1))
        print(f'Test Loss: {test_loss.item()}')
        wandb.log({'Test Loss': test_loss.item()})
    
    test_predictions_unscaled = reverse_scale_preds(test_predictions)
    y_test_unscaled = joblib.load('./data_processing/y_test_unscaled.gz')

    agg_test_preds = average_overlapping_series(test_predictions_unscaled)
    agg_test_actuals = average_overlapping_series(y_test_unscaled)

    # plot predicted vs. actuals 
    plt.figure(figsize=(10, 6))
    plt.plot(agg_test_actuals, label='Actual')
    plt.plot(agg_test_preds, label='Predicted')
    plt.title('Test Predictions vs Actual')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Wave Height (Meters)')
    plt.legend()

    fig_str = "../results/figures/test_" + config["model_name"] + ".png"
    plt.savefig(fig_str, format='png', dpi=200) 

    # log the plot image file to wandb
    wandb.log({"Predictions vs Actual": wandb.Image(fig_str)})

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

def reverse_scale_preds(test_predictions):
    scaler = joblib.load('./data_processing/scaler.gz') 

    # turn to numpy array
    test_predictions_np = test_predictions.numpy()
    
    # reshape into 2d arr w/ 1 col
    # -1 to calc the necessary number of rows to maintain the same total number of elements 
    test_predictions_np_reshaped = test_predictions_np.reshape(-1, 1)

    # same shape as test_predictions_np_reshaped but filled with zeros
    dummy_feature = np.zeros_like(test_predictions_np_reshaped)

    # horizontally stack the dummy_feature array and test_predictions_np_reshaped array into a single 2d array   
    test_predictions_combined = np.hstack((dummy_feature, test_predictions_np_reshaped))

    # once of the shape expected by the scaler, do reverse scaling 
    test_predictions_unscaled_combined = scaler.inverse_transform(test_predictions_combined)

    # extract column with unscaled values 
    test_predictions_unscaled = test_predictions_unscaled_combined[:, 1]

    # reshape into the original shape: (n_samples, sequence_next)
    test_predictions_unscaled = test_predictions_unscaled.reshape(int(test_predictions_unscaled.shape[0] / config["sequence_next"]), config["sequence_next"])

    return test_predictions_unscaled
