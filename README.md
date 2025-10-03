# Predicting Wave Height at Morro Bay with LSTM Neural Networks

## Demo

[Google Slides Link](https://docs.google.com/presentation/d/149-wIPDYxhoVXCVjdC5OUCRpQVXRgJ-vjqI0PdGPWM4/edit?usp=sharing)

## Description

This project uses a long short-term memory (LSTM) neural network to predict surf report data. Using hourly data from Morro Bay provided by Surfline, the model predicts surf forecasts based on various input variables, such as buoy wave height, tide, swell period, and beach wave height. The project explores different predictor-target pairs to identify the best predictors for key metrics like wave height at the beach. Model performance is evaluated with unseen data using mean-squared error (MSE) and other metrics. This project demonstrates AI's potential in environmental modeling and offers valuable insights to a broad range of people, spanning from surfers to data scientists. 

Cal Poly San Luis Obispo - Computer Science and Software Engineering Department

Advisor – Franz Kurfess 

## How to setup
1. Clone repo
2. See requirements.txt for a list the necessary Python packages
3. Contact me for:
  - the Surfline CSV data
  - access to [The WandB project](https://wandb.ai/qap-ai/Surf%20Forecast%20AI?nw=nwuserqap2001)

## How to run
`python main.py`

- key parameter notes:
  - to run sweeps, turn `sweep = True` in main.py
  - to set sweep and model parameters, see config.py 

## Tools used during development
- VS Code 
- Github 
- [Weights and Biases](https://wandb.ai/site) (WandB) 
- Python

## Contact Information
qap2001@gmail.com

