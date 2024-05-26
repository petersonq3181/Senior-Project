config = {
    "model_name": "LSTM_model3_sweeps0", 
    "MSELoss_criterion": "mean",
    "sequence_lag": 48,
    "sequence_delay": 1,
    "sequence_next": 12,
    "batch_size": 64,
    "input_dim": 6,
    "hidden_dim": 2,
    "layer_dim": 1,
    "learning_rate": 0.05,
    "epochs": 10,
    "patience": 10,
    "min_delta": 0.001,
    "n_iters_eval": 100
}

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "loss",
        "goal": "minimize"   
    },
    "parameters": {
        "sequence_lag": {
            "values": [12, 48]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "learning_rate": {
            "values": [0.001, 0.01, 0.05, 0.1]
        }
    }
}

