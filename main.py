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
# print(type(df), df)

# make sure to standardize units to meters 
df['lotusMaxBWH_ft'] = df['lotusMaxBWH_ft'] * 0.3048 


train = df.iloc[:30000]
test = df.iloc[30000:]

scaler = MinMaxScaler()
scaler.fit(train[numeric_cols])
scaled_train = scaler.transform(train[numeric_cols])
scaled_test = scaler.transform(test[numeric_cols])

# if need to keep the datetime information alongside the scaled data, can insert the scaled data back into a DataFrame
scaled_train_df = pd.DataFrame(scaled_train, columns=numeric_cols, index=train.index)
scaled_train_df['datetime_local'] = train['datetime_local']

scaled_test_df = pd.DataFrame(scaled_test, columns=numeric_cols, index=test.index)
scaled_test_df['datetime_local'] = test['datetime_local']


print(scaled_train[:10])
print(scaled_test[:10])


# Define a function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length):
        x = data[i:(i+sequence_length)]
        y = data[i+1:(i+1+sequence_length)]
        xs.append(x)
        ys.append(y[-1])
    return np.array(xs), np.array(ys)

sequence_length = 14  # Same as n_input in your initial approach
batch_size = 64


# Prepare data
X_train, y_train = create_sequences(scaled_train_df['lotusMaxBWH_ft'].values, sequence_length)
X_test, y_test = create_sequences(scaled_test_df['lotusMaxBWH_ft'].values, sequence_length)

# Convert to tensors
X_train_tensor = torch.tensor(X_train).float().unsqueeze(-1)  # Adding required extra dimension
y_train_tensor = torch.tensor(y_train).float()
X_test_tensor = torch.tensor(X_test).float().unsqueeze(-1)
y_test_tensor = torch.tensor(y_test).float()

# DataLoader setup
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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

model = LSTMModel(input_dim=1, hidden_dim=100, num_layers=1, output_dim=1)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 20
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

