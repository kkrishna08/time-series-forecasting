# -*- coding: utf-8 -*-
"""Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 2. Load and Preprocess the Data
# (Assuming data is a NumPy array or Pandas DataFrame containing the time series data)
# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

# Inspect the dataset
print(data.head())
print("\nData Summary:")
print(data.describe())

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data['Temp'])
plt.title('Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# 1. Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data['Temp'].values.reshape(-1, 1))

# 2. Define Sequence Length and Forecasting Horizon
sequence_length = 10  # e.g., using the past 10 days to predict the next day
forecast_horizon = 1  # predict the temperature for the next day

# 3. Create Sequences for Supervised Learning
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+forecast_horizon-1])
    return np.array(X), np.array(y)

X, y = create_sequences(data_normalized, sequence_length, forecast_horizon)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Output the shapes of the training and test sets to verify
print(f"Training set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test set shape: X_test = {X_test.shape}, y_test = {y_test.shape}")
# Normalize the dataset
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data['Temp'].values.reshape(-1, 1))  # Access the 'Temp' column as a NumPy array using .values

# Define sequence length and forecasting horizon
sequence_length = 10  # e.g., 10 previous time steps
forecast_horizon = 1  # e.g., predict the next time step

# Function to create sequences
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+forecast_horizon-1])
    return np.array(X), np.array(y)

# Prepare data
X, y = create_sequences(data_normalized, sequence_length, forecast_horizon)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Define Model Architectures

# RNN Model
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# GRU Model
def build_gru_model(input_shape):
    model = Sequential([
        GRU(50, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Set input shape
input_shape = (sequence_length, 1)

# Initialize models
rnn_model = build_rnn_model(input_shape)
lstm_model = build_lstm_model(input_shape)
gru_model = build_gru_model(input_shape)

# Print model summaries
print("RNN Model Summary:")
rnn_model.summary()
print("\nLSTM Model Summary:")
lstm_model.summary()
print("\nGRU Model Summary:")
gru_model.summary()

# 4. Model Training
epochs = 5
batch_size = 32

# Train RNN
history_rnn = rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Train LSTM
history_lstm = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Train GRU
history_gru = gru_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# 5. Model Evaluation
rnn_eval = rnn_model.evaluate(X_test, y_test)
lstm_eval = lstm_model.evaluate(X_test, y_test)
gru_eval = gru_model.evaluate(X_test, y_test)

print(f"\nRNN Model Test Loss: {rnn_eval}")
print(f"LSTM Model Test Loss: {lstm_eval}")
print(f"GRU Model Test Loss: {gru_eval}")

# 6. Performance Comparison
# Plot training and validation loss
plt.figure(figsize=(12, 6))

# Plot RNN Loss
plt.plot(history_rnn.history['loss'], label='RNN Training Loss')
plt.plot(history_rnn.history['val_loss'], label='RNN Validation Loss')

# Plot LSTM Loss
plt.plot(history_lstm.history['loss'], label='LSTM Training Loss')
plt.plot(history_lstm.history['val_loss'], label='LSTM Validation Loss')

# Plot GRU Loss
plt.plot(history_gru.history['loss'], label='GRU Training Loss')
plt.plot(history_gru.history['val_loss'], label='GRU Validation Loss')

plt.title('Training and Validation Loss by Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()