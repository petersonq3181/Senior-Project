import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time 

     

fn = 'MorroBayHeights.csv'
cols = ['datetime', 'lotusSigh_mt', 'datetime_local', 'lotusMaxBWH_ft']
numeric_cols = ['lotusSigh_mt', 'lotusMaxBWH_ft']
df = pd.read_csv(fn, usecols=cols, parse_dates=['datetime_local'])

# make sure to standardize units to meters 
df['lotusMaxBWH_ft'] = df['lotusMaxBWH_ft'] * 0.3048 

# splitting to approximately 80 20 train test respectively 
# 37266 total 
train = df.iloc[:29800]
test = df.iloc[29800:]

# ----- parameters 
sequence_length = 12
batch_size = 64

input_dim = 1
output_dim = 1
hidden_dim = 4
layer_dim = 1
learning_rate = 0.03
epochs = 10

# ----- preprocessing 
# scale each numerical feature into a range [0 1] (scaling based on only the training data)
# then applies same scaling to the test set 
scaler = MinMaxScaler()
scaler.fit(train[numeric_cols])
scaled_train = scaler.transform(train[numeric_cols])
scaled_test = scaler.transform(test[numeric_cols])

rev_scaled_test = np.array(scaler.inverse_transform(scaled_test))

# ----- data setup 
def create_sequences(input_data, target_data, seq_len):
    xs = []
    ys = []
    for i in range(len(input_data) - seq_len):
        x = input_data[i:(i + seq_len)]  
        y = target_data[i + seq_len]  
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# assuming scaled_train[:, 0] is A and scaled_train[:, 1] is B
X_train, y_train = create_sequences(scaled_train[:, 0].reshape(-1, 1), scaled_train[:, 1], sequence_length)
X_test, y_test = create_sequences(scaled_test[:, 0].reshape(-1, 1), scaled_test[:, 1], sequence_length)

# convert to tensors without adding extra dimension
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).float()

# DataLoader setup
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ----- model 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        # hidden dimensions
        self.hidden_dim = hidden_dim

        # number of hidden layers
        self.layer_dim = layer_dim
        
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # readout layer 
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)

        # init hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()

        # init cell state 
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()

        # need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # just want the last time stamp's hidden state 
        out = self.fc(out[:, -1, :])

        return out


# ----- training 
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_test_loss = float('inf')

# number of evaluations to wait for improvement
patience = 10
# counter to track evaluations without improvement
patience_counter = 0
min_delta = 0.001 
n_iters_eval = 100

iter = 0
start_time = time.time()
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        # ensure model is in training mode 
        model.train() 
        iter += 1 

        # forward pass
        y_pred = model(x_batch).squeeze(-1)
        loss = criterion(y_pred, y_batch)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % n_iters_eval == 0:
            # set the model to evaluation mode
            model.eval()  

            test_losses = []
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    y_test_pred = model(x_test).squeeze(-1)
                    test_loss = criterion(y_test_pred, y_test)
                    test_losses.append(test_loss.item())
            
            avg_test_loss = sum(test_losses) / len(test_losses)
            
            print(f'\tEpoch [{epoch+1}/{epochs}], Step [{iter}], Train Loss: {loss.item():.6f}, Test Loss: {avg_test_loss:.6f}')

            if avg_test_loss < (best_test_loss - min_delta):
                best_test_loss = avg_test_loss

                # save the best model so far 
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print('early stopping triggered')

                # load the best model parameters before stopping
                model.load_state_dict(torch.load('best_model.pth'))
                break

    if patience_counter >= patience:
        break 

    print(f'Epoch {epoch+1},\t Loss: {loss.item()}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

# ----- testing 
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)

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
