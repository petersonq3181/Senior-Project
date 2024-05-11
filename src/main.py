from data_processing.preprocess import preprocess_data
from train import train_lstm
from test import test_lstm 
from config import config, sweep_config
import wandb

# init a sweep
sweep_id = wandb.sweep(sweep_config, project="Surf Forecast AI")

def main():

    for i in range(1):
        # init Weights and Biases run 
        wandb.init(
            project = "Surf Forecast AI",
            name = config["model_name"] + "_run_" + str(i),
            config = config
        )

        config.update({
            "sequence_lag": wandb.config.sequence_lag,
            "batch_size": wandb.config.batch_size,
            "learning_rate": wandb.config.learning_rate
        })

        # data preprocessing 
        x_train, y_train, x_test, y_test = preprocess_data("../data/raw/MorroBayHeights.csv")
    
        # train 
        model = train_lstm(x_train, y_train)

        # test 
        test_lstm(model, x_test, y_test)

        wandb.finish()

if __name__ == "__main__":
    wandb.agent(sweep_id, main)
