from torch.utils.data import TensorDataset, DataLoader
from model import LSTMModel
from config import config
import torch
import time 
import wandb

def train_lstm(x_train_tensor, y_train_tensor):
    
    # print('shape of x_train_tensor:')
    # print(x_train_tensor.shape)
    # print('shape of y_train_tensor:')
    # print(x_train_tensor.shape)
    # print()

    # split data into training and validation sets (80/20 training validation subset split)
    total_samples = x_train_tensor.shape[0]
    split_index = int(total_samples * 0.8)
    x_train = x_train_tensor[:split_index]
    y_train = y_train_tensor[:split_index]
    x_validate = x_train_tensor[split_index:]
    y_validate = y_train_tensor[split_index:]

    # data setup
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    validate_dataset = TensorDataset(x_validate, y_validate)
    validate_loader = DataLoader(validate_dataset, batch_size=config["batch_size"], shuffle=False)

    model = LSTMModel(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], layer_dim=config["layer_dim"], output_dim=config["sequence_next"])
    criterion = torch.nn.MSELoss(reduction=config["MSELoss_criterion"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_valid_loss = float('inf')
    best_loss = float('inf')
 
    iter = 0
    start_time = time.time()
    for epoch in range(config["epochs"]):
        print('ITER: ', iter)
        for x_batch, y_batch in train_loader:
            # ensure model is in training mode 
            model.train() 
            iter += 1 

            # forward pass
            y_pred = model(x_batch).squeeze(-1)
            y_batch = y_batch.squeeze(-1) # squeeze last dim; from (batch_size, sequence_next, 1) --> (batch_size, sequence_next)

            loss = criterion(y_pred, y_batch)
            
            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"Training Loss": loss.item()})

            if iter % config["n_iters_eval"] == 0:
                # set the model to evaluation mode
                model.eval()  

                validate_losses = []
                with torch.no_grad():
                    for x_validate, y_validate in validate_loader:
                        y_pred = model(x_validate).squeeze(-1)
                        y_validate = y_validate.squeeze(-1) # squeeze last dim; from (batch_size, sequence_next, 1) --> (batch_size, sequence_next)
                        test_loss = criterion(y_pred, y_validate)
                        validate_losses.append(test_loss.item())
                
                avg_test_loss = sum(validate_losses) / len(validate_losses)
                
                print(f'\tEpoch [{epoch+1}/{config["epochs"]}], Step [{iter}], Train Loss: {loss.item():.6f}, Test Loss: {avg_test_loss:.6f}')

                model_str = "../results/models/" + config["model_name"] + ".pth"
                if avg_test_loss < (best_valid_loss - config["min_delta"]):
                    best_valid_loss = avg_test_loss

                    # save the best model so far 
                    torch.save(model.state_dict(), model_str)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config["patience"]:
                    print('early stopping triggered')

                    # load the best model parameters before stopping
                    model.load_state_dict(torch.load(model_str))
                    break

        if loss.item() < best_loss:
            best_loss = loss.item()
            wandb.log({"best_loss": best_loss})

        print(f'Epoch {epoch+1},\t Loss: {loss.item()}')

    # at the end of training, log the final or best loss as a summary
    wandb.run.summary["loss"] = best_loss

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")
    wandb.log({"Training Time (s)": elapsed_time}) 

    return model
