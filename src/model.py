import torch.nn as nn
import torch 

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
    