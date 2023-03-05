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
from keras.layers import Dense, Dropout, LSTM, LeakyReLU, Activation#, GRU
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_ohlc(crypt):
    crypt_name = crypt + '-USD'
    temp = yf.Ticker(crypt_name)
    data = temp.history(period = 'max', interval="1d")
    return data

def transform_data():
    # GET BTC data
    df = get_ohlc('BTC')
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df, scaler

# create a function to prepare the data for input into the LSTM model
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix >= len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_data(X,y):
    # Split the data into training and testing sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f'length of traininig set: {len(X_train)}')
    print(f'length of test set: {len(X_test)}')
    return X_train, y_train, X_test, y_test
    # train_pct = int(len(df)*.75)
    # training_set = df['Close'].iloc[:train_pct].values
    # testing_set = df['Close'].iloc[train_pct:].values
    # # Define the input and output variables
    # X_train = []
    # y_train = []
    # X_test = []
    # y_test = []

def algorithm(X_train, y_train, X_test, y_test, n_steps):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation=LeakyReLU(alpha=0.1), return_sequences=True, input_shape=(n_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation=LeakyReLU(alpha=0.1), return_sequences=True, input_shape=(n_steps, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation=LeakyReLU(alpha=0.1), return_sequences=True, input_shape=(n_steps, 1)))
    model.add(Dropout(0.2))
    # model.add(LSTM(50, activation=LeakyReLU(alpha=0.1), return_sequences=True, input_shape=(n_steps, 1)))
    # model.add(Dropout(0.2))
    model.add(LSTM(50, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.summary()
    # compile the model
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
    # fit the model to the training data
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2,
                         validation_data=(X_test, y_test),callbacks=[early_stop])
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training loss',color='r')
    plt.plot(history.history['val_loss'], label='Validation loss',color='b')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('train_val_loss.png',dpi=300)
    # make predictions on the test data
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    print('RMSE error:', rmse)
    return model

def predict_future(df,n_steps,model,scaler):
    # use the model to predict the next 7 days of prices
    last_30_days = df['Close'].tail(n_steps).values.reshape(-1, 1)
    next_7_days = []
    for i in range(7):
        pred = model.predict(last_30_days.reshape(1, n_steps, 1))
        next_7_days.append(pred[0][0])
        last_30_days = np.concatenate((last_30_days[1:], pred), axis=0)
    # print the predicted prices for the next 7 days
    temp_arr = np.array(next_7_days).reshape(-1, 1)
    print('Predicted prices for the next 7 days:', scaler.inverse_transform(temp_arr))
    df['Close'] = scaler.inverse_transform(df['Close'].values.reshape(-1,1))
    return df, scaler.inverse_transform(temp_arr)

def plot_data(df,next_7_days):
    start_date = datetime.fromtimestamp(df.index[-1].timestamp())
    date_list = [start_date + timedelta(days=x) for x in range(7)]
    plt.figure(figsize=(15,8))
    plt.plot(df.tail(120).index,df['Close'].tail(120).values,color='b',marker='*',label='Past')
    plt.plot(date_list,next_7_days,color='r',marker='*',label='Future')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('final_prediction.png',dpi=350)

def plot_data_multiple(df,save_forecasts):
    start_date = datetime.fromtimestamp(df.index[-1].timestamp())
    date_list = [start_date + timedelta(days=x) for x in range(7)]
    plt.figure(figsize=(15,8))
    plt.plot(df.tail(120).index,df['Close'].tail(120).values,color='b',marker='*',label='Past')
    for i in range(0,5):
        label_val = 'Future_' + str(i)
        plt.plot(date_list,save_forecasts[i],marker='*',label=label_val)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    # 
# for i in range(60, 2310):
#     X_train.append(training_set[i-60:i])
#     y_train.append(training_set[i])
# for i in range(60, 770):
#     X_test.append(testing_set[i-60:i])
#     y_test.append(testing_set[i])
# X_train, y_train = np.array(X_train), np.array(y_train)

# # Reshape the input data
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32)

# # Prepare the input data for testing
# inputs = df['Close'].iloc[len(df) - len(testing_set) - 60:, 1:2].values
# inputs = scaler.transform(inputs)

# # Reshape the input data
# X_test = []
# for i in range(60, 343):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Make predictions on the test data
# predicted_price = model.predict(X_test)
# predicted_price = scaler.inverse_transform(predicted_price)

# # Evaluate the model
# rmse = np.sqrt(np.mean((predicted_price - testing_set) ** 2))
# print('Root Mean Squared Error:', rmse)

# # Predict the Bitcoin price 4 days from now
# last_4_days = df.iloc[-60:, 1:2].values
# last_4_days = scaler.transform(last_4_days)
# X_pred = np.array([last_4_days])
# X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
# predicted_price_4_days = model.predict(X_pred)
# predicted_price_4_days = scaler.inverse_transform(predicted_price_4_days)
# print('Predicted Bitcoin price 4 days from now:', predicted_price_4_days)

def main():
    df, scaler = transform_data()
    # split_data(df)
    n_steps = 90 #Increasing the sequence length allows the model to capture more long-term dependencies in the data, but can also increase the computational cost of training the model.
    # prepare the data for input into the LSTM model
    X, y = prepare_data(df['Close'].values, n_steps)
    X_train, y_train, X_test, y_test = split_data(X,y)
    save_forecasts = []
    for i in range(0,5):
        model_val = algorithm(X_train, y_train, X_test, y_test, n_steps)
        df, future_vals = predict_future(df,n_steps,model_val,scaler)
        save_forecasts.append(future_vals)
    plot_data_multiple(df,save_forecasts)
    # plot_data(df,future_vals)
    # print(X)
    # print(y)
if __name__ == "__main__":
    main()