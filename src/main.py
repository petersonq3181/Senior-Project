from data_processing.preprocess import preprocess_data
from train import train_lstm
from test import test_lstm 
from config import config, sweep_config
import wandb
import time

sweep = True 

start_time = time.time()

if sweep:
    # init a sweep
    sweep_id = wandb.sweep(sweep_config, project="Surf Forecast AI")

def main():
    
    if sweep: 
        # init Weights and Biases run 
        wandb.init(
            project = "Surf Forecast AI",
            name = config["model_name"] + "_run",
            config = config
        )

        config.update({
            "sequence_lag": wandb.config.sequence_lag,
            "batch_size": wandb.config.batch_size,
            "learning_rate": wandb.config.learning_rate
        })

        # data preprocessing 
        x_train, y_train, x_test, y_test = preprocess_data("../data/MorroBayHeights.csv")

        # train 
        model = train_lstm(x_train, y_train)

        # test 
        test_lstm(model, x_test, y_test)

        wandb.finish()

    else: 
        # data preprocessing 
        x_train, y_train, x_test, y_test = preprocess_data("../data/MorroBayHeights.csv")

        for i in range(1):
            # init Weights and Biases run 
            wandb.init(
                project = "Surf Forecast AI",
                name = config["model_name"] + "_run_" + str(i),
                config = config
            )

            # train 
            model = train_lstm(x_train, y_train)

            # test 
            test_lstm(model, x_test, y_test)

            wandb.finish()

if __name__ == "__main__":
    if sweep: 
        wandb.agent(sweep_id, main)
    else:
        main()

    print(f"Time to execute: {time.time() - start_time} seconds")
