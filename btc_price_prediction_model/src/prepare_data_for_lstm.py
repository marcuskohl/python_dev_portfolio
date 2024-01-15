import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        label = target[i + sequence_length]
        X.append(sequence)
        y.append(label)
    return np.array(X), np.array(y)

def main():
    #Loading dataset
    file_path = '../data/processed_BTCUSDT_Hourly.csv'
    btc_data = pd.read_csv(file_path)
    
    #Dropping 'datetime' as a feature
    btc_data_features = btc_data.drop(columns=['datetime'])
    
    #Using 'close' as target
    target_column = 'close'
    features = btc_data_features.drop(columns=[target_column])
    target = btc_data_features[target_column]

    #Normalizing features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_normalized = scaler.fit_transform(features)
    target_normalized = scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    #Defining sequence length
    sequence_length = 100  

    #Creating sequences
    X, y = create_sequences(features_normalized, target_normalized, sequence_length)

    #Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    #Saving processed data and scaler object for future use
    np.save('../data/X_train.npy', X_train)
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/y_test.npy', y_test)
    with open('../data/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    print("Complete. Train and test sets are saved.")

if __name__ == '__main__':
    main()





