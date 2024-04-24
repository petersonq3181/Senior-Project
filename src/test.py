import matplotlib.pyplot as plt
import numpy as np
import config
import torch



def test_lstm(model, x_test_tensor, y_test_tensor):
    model.eval()

    criterion = torch.nn.MSELoss(reduction=config.MSELoss_criterion)

    with torch.no_grad():
        test_predictions = model(x_test_tensor)

        test_predictions = test_predictions.squeeze(-1)
        
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item()}')
