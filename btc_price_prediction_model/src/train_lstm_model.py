import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Loading training data
X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')
X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')

#Defining LSTM structure
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#Compiling model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

#Defining callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('../models/best_btc_price_prediction_model.keras', save_best_only=True, verbose=1)

#Training model
history = model.fit(
X_train, y_train,
epochs=100,
batch_size=32,
validation_data=(X_test, y_test),
callbacks=[early_stopping, model_checkpoint],
verbose=1
)

#Saving model
model.save('../models/btc_price_prediction_model.keras')

print("Training complete and model saved.")




