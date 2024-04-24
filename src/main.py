from data_processing.preprocess import preprocess_data


def main():
    # data preprocessing 
    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = preprocess_data('../raw/MorroBayHeights.csv')

    # train 
    

if __name__ == "__main__":
    main()
