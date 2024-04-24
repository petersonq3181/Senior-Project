from data_processing.preprocess import preprocess_data
from train import train_lstm
from test import test_lstm 


def main():
    # data preprocessing 
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data('../data/raw/MorroBayHeights.csv')

    # train 
    model = train_lstm(x_train_tensor, y_train_tensor)

    # test 
    test_lstm(model, x_test_tensor, y_test_tensor)

if __name__ == "__main__":
    main()
