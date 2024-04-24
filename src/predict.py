import matplotlib.pyplot as plt
import numpy as np
import torch



def test_lstm(model, x_test_tensor, y_test_tensor):
    model.eval()

    with torch.no_grad():
        test_predictions = model(x_test_tensor)

        test_predictions = test_predictions.squeeze(-1)
        
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item()}')

    y_test = rev_scaled_test[:, 1]

    test_predictions_np = test_predictions.numpy()
    test_predictions_np_reshaped = test_predictions_np.reshape(-1, 1)
    dummy_feature = np.zeros_like(test_predictions_np_reshaped)
    test_predictions_combined = np.hstack((dummy_feature, test_predictions_np_reshaped))
    test_predictions_unscaled_combined = scaler.inverse_transform(test_predictions_combined)
    test_predictions_unscaled = test_predictions_unscaled_combined[:, 1]

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(test_predictions_unscaled, label='Predicted')
    plt.title('Test Predictions vs Actual')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Wave Height (Meters)')
    plt.legend()
    plt.show()

    # plotting 
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Loss')
    plt.title('Training MSE')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
