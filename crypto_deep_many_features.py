import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import ta
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os

class BitcoinPricePredictor:
    def __init__(self, crypt, n_features, n_steps, n_outputs, n_epochs, batch_size):
        self.crypt_name = crypt
        self.n_features = n_features
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        crypt_name = crypt + '-USD'
        temp = yf.Ticker(crypt_name)
        self.data = temp.history(period = 'max', interval="1d")

    def RSI_MACD_calc(self):
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close'], window=14,fillna=True).rsi()
        macd = ta.trend.macd(self.data['Close'],fillna=True)
        # Add MACD and signal lines to the DataFrame
        self.data['MACD'] = macd
        self.data['OBV'] = ta.volume.OnBalanceVolumeIndicator(self.data['Close'], self.data['Volume']).on_balance_volume()
        aroon = ta.trend.AroonIndicator(self.data['Close'],window=25,fillna=True)
        self.data['aroon'] = aroon.aroon_indicator()
        mfi = ta.volume.MFIIndicator(high=self.data['High'],low=self.data['Low'],close=self.data['Close'],
                                     volume=self.data['Volume'],fillna=True)
        self.data['MFI'] = mfi.money_flow_index()
        vwprice = ta.volume.VolumeWeightedAveragePrice(high=self.data['High'],low=self.data['Low'],close=self.data['Close'],volume=self.data['Volume'],fillna=True)
        self.data['VMAP'] = vwprice.volume_weighted_average_price()
        

    def prepare_data(self, data):
        # Extract relevant features
        features = ['Close', 'Low', 'High', 'MACD', 'RSI', 'OBV','aroon','Volume','MFI','VMAP']
        data = self.data[features]

        # Scale data
        self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        # Scale data
        data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)
        # data_close = data[['Close']]
        data_close = self.scaler1.fit_transform(data_close)
        data_non_close = data[['Low', 'High', 'MACD', 'RSI','OBV','aroon','Volume','MFI','VMAP']]
        data_non_close = self.scaler2.fit_transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

        # Split data into input/output sequences
        X, y = [], []
        for i in range(len(data)-self.n_steps-self.n_outputs+1):
            X.append(data[i:i+self.n_steps, :])
            y.append(data[i+self.n_steps:i+self.n_steps+self.n_outputs, 0])
        X, y = np.array(X), np.array(y)

        # Split data into training/validation sets
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        return X_train, y_train, X_val, y_val

    def create_model(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.9
        )
        drop_val = 0.25
        model = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9, activation='relu',return_sequences=True, input_shape=(self.n_steps, self.n_features))),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(drop_val),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu',return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(7, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, activation='relu')),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.n_outputs,activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss='mse')
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        if os.path.exists(f"{self.crypt_name}_model.h5"):
            self.model = load_model(f"{self.crypt_name}_model.h5")
        else:
            model = self.create_model()
            es = tf.keras.callbacks.EarlyStopping("val_loss",patience=40)
            model.fit(X_train, y_train, 
                      epochs=self.n_epochs, 
                      batch_size=self.batch_size, 
                      validation_data=(X_val, y_val),
                      callbacks=[es],
                      verbose=2)
            model.summary()
            self.model = model
            self.model.save(f"{self.crypt_name}_model.h5")

    def predict(self, data):
        # Prepare data for prediction
        data = data[['Close', 'Low', 'High', 'MACD', 'RSI','OBV','aroon','Volume','MFI','VMAP']]
        # data_close = data[['Close']]
        data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)
        data_non_close = data[['Low', 'High', 'MACD', 'RSI','OBV','aroon','Volume','MFI','VMAP']]
        data_close = self.scaler1.transform(data_close)
        data_non_close = self.scaler2.transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)
        X_pred = np.array([data[-self.n_steps:, :]])

        # Make prediction
        y_pred = self.model.predict(X_pred)
        y_pred = self.scaler1.inverse_transform(y_pred)[0]
        print(y_pred)
        # # Prepare data for prediction
        # data = data[['Close', 'Low', 'High', 'MACD', 'RSI']]
        # data = self.scaler.transform(data)
        # X_pred = np.array([data[-self.n_steps:, :]])
        # X_pred = np.reshape(X_pred, (1, self.n_steps, self.n_features))
        # # Make prediction
        # y_pred = self.model.predict(X_pred)
        # print(y_pred)
        # y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.n_features))[0]  
        # # y_pred = y_pred[:, 0] # extract only the predicted Close prices

        return y_pred, self.data.index[-1]
    
    def plot_results(self):
        pred = pd.read_csv(f'{self.crypt_name}_pred.csv')
        # plt.plot(self.data['Close'], label='Actual')
        plt.plot(pred['date'],pred['pred'], label='Predicted')
        plt.title(f'{self.crypt_name} Close Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    def run_analysis(self):
        #calculate MACD and RSI
        self.RSI_MACD_calc()
        # Prepare data for training
        X_train, y_train, X_val, y_val = self.prepare_data(self.data)
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        # Make prediction for the next 30 days
        prediction, last_date = self.predict(self.data)
        print(pd.to_datetime(last_date))
        start_date = pd.to_datetime(last_date).date() + timedelta(days=1)
        end_date = start_date + timedelta(days=len(prediction)-1)
        date_range_array = pd.date_range(start=start_date, end=end_date)
        pd.DataFrame({'pred':prediction,'date':date_range_array}).to_csv(f'{self.crypt_name}_pred.csv',index=False)
        self.plot_results()

BitcoinPricePredictor(crypt="BTC",
                    n_features=10, 
                    n_steps=60, 
                    n_outputs=30, 
                    n_epochs=250, 
                    batch_size=64).run_analysis()
