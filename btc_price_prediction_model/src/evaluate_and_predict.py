import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_model(model_path):
    """Loading and returning trained LSTM model."""
    return tf.keras.models.load_model(model_path)

def load_test_data(X_path, y_path):
    """Loading test dataset."""
    X_test = np.load(X_path)
    y_test = np.load(y_path)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluating the model on the test set."""
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss (Mean Squared Error): {loss}")

def make_predictions(model, X_test):
    """Using the model to make predictions."""
    predictions = model.predict(X_test)
    return predictions.flatten() 

def plot_predictions(y_test, predictions):
    """Plotting actual vs. predicted values."""
    plt.figure(figsize=(15, 7))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='orange', linestyle='--')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    model_path = '../models/best_btc_price_prediction_model.keras' 
    X_test_path = '../data/X_test.npy'
    y_test_path = '../data/y_test.npy'
    scaler_path = '../data/scaler.pkl'

    model = load_model(model_path)
    X_test, y_test = load_test_data(X_test_path, y_test_path)
    evaluate_model(model, X_test, y_test)

    predictions = make_predictions(model, X_test)

    #Loading scaler object
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    #Rescaling predictions and y_test back to original price scale
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    #Evaluating model performance
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    mean_actual_price = np.mean(y_test_rescaled)
    mae_percentage = (mae / mean_actual_price) * 100
    rmse_percentage = (rmse / mean_actual_price) * 100

    #Printing performance metrics
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"MAE as a percentage of mean actual price: {mae_percentage:.2f}%")
    print(f"RMSE as a percentage of mean actual price: {rmse_percentage:.2f}%")

    #Plotting actual vs. predicted prices
    plot_predictions(y_test_rescaled, predictions_rescaled)

if __name__ == '__main__':
    main()


    


