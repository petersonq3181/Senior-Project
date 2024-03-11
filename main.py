import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

     

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
num_layers = 1
learning_rate = 0.03
epochs = 10

# ----- preprocessing 
# scale each numerical feature into a range [0 1] (scaling based on only the training data)
# then applies same scaling to the test set 
scaler = MinMaxScaler()
scaler.fit(train[numeric_cols])
scaled_train = scaler.transform(train[numeric_cols])
scaled_test = scaler.transform(test[numeric_cols])


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


# ----- model 
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out


# ----- training 
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)

        # squeeze the predictions to match the target's shape
        y_pred = y_pred.squeeze(-1)  
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1},\t Loss: {loss.item()}')


# ----- testing 
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)
    
    test_predictions = test_predictions.squeeze(-1)
    
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

test_predictions_np = test_predictions.numpy() * scaler.scale_[1] + scaler.min_[1]
y_test_np = y_test_tensor.numpy() * scaler.scale_[1] + scaler.min_[1]

# plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions_np, label='Predicted')
plt.title('Test Predictions vs Actual')
plt.xlabel('Time')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

