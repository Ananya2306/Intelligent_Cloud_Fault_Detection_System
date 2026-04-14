import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(data):
    print("Training LSTM...")

    # Use small sample
    data = data.head(5000)

    X = data[['cpu','memory','max_usage']].values
    y = data['fault'].values

    # reshape for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(1,3)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X, y, epochs=5, verbose=1)

    model.save("models/lstm_model.h5")

    print("LSTM model trained and saved!")

    return model