from torch.utils.data import TensorDataset, DataLoader
from utils import LSTMModel
import torch
import time 

def train_lstm(x_train_tensor, y_train_tensor):
    # data setup
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
    train_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        print('ITER: ', iter)
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

            train_losses.append(loss.item())

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

    return model
