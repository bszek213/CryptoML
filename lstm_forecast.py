#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM price predictions
Should this be on the price or the cumulative log transform?
@author: bszekely
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def get_ohlc(crypt):
    crypt_name = crypt + '-USD'
    temp = yf.Ticker(crypt_name)
    data = temp.history(period = 'max', interval="1d")
    return data

# GET BTC data
df = get_ohlc('BTC')
# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# Split the data into training and testing sets
train_pct = int(len(df)*.75)
training_set = df['Close'].iloc[:train_pct].values
testing_set = df['Close'].iloc[train_pct:].values
print(f'length of traininig set: {len(training_set)}')
print(f'length of test set: {len(testing_set)}')
# Define the input and output variables
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(60, 2310):
    X_train.append(training_set[i-60:i])
    y_train.append(training_set[i])
for i in range(60, 770):
    X_test.append(testing_set[i-60:i])
    y_test.append(testing_set[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare the input data for testing
inputs = df['Close'].iloc[len(df) - len(testing_set) - 60:, 1:2].values
inputs = scaler.transform(inputs)

# Reshape the input data
X_test = []
for i in range(60, 343):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test data
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

# Evaluate the model
rmse = np.sqrt(np.mean((predicted_price - testing_set) ** 2))
print('Root Mean Squared Error:', rmse)

# Predict the Bitcoin price 4 days from now
last_4_days = df.iloc[-60:, 1:2].values
last_4_days = scaler.transform(last_4_days)
X_pred = np.array([last_4_days])
X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
predicted_price_4_days = model.predict(X_pred)
predicted_price_4_days = scaler.inverse_transform(predicted_price_4_days)
print('Predicted Bitcoin price 4 days from now:', predicted_price_4_days)
