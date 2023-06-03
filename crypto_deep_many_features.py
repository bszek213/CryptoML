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
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from colorama import Fore, Style
from sys import argv

class changePricePredictor:
    def __init__(self, crypt, n_features, n_steps, n_outputs, n_epochs, batch_size):
        self.crypt_name = crypt
        # self.n_features = n_features
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        crypt_name = crypt + '-USD'
        temp = yf.Ticker(crypt_name)
        price_data = temp.history(period = 'max', interval="1d")
        print(Fore.GREEN,f'NUMBER OF SAMPLES FOR {crypt_name}: {len(price_data)}',Style.RESET_ALL)
        self.features = ['Close', 'Low', 'High', 'momentum_stoch_rsi', 'trend_aroon_down', 'volume_vpt', 'volume_em', 'trend_aroon_up', 'trend_macd_diff', 'volume_obv']
        self.non_close_features = ['Low', 'High', 'momentum_stoch_rsi', 'trend_aroon_down', 'volume_vpt', 'volume_em', 'trend_aroon_up', 'trend_macd_diff', 'volume_obv']
        self.n_features = len(self.features)
        self.data = ta.add_all_ta_features(
            price_data,
            open="Open",
            high="High",
            close='Close',
            low='Low',
            volume='Volume',
            fillna=True
        )
        # print(self.data.columns)

    def prepare_data(self, data):
        # Extract relevant features
        data = self.data[self.features]

        # Scale data
        # self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        # Scale data
        data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)
        # data_close = data[['Close']]
        # data_close = self.scaler1.fit_transform(data_close)
        data_non_close = data[self.non_close_features]
        data_non_close = self.scaler2.fit_transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

        # Split data into input/output sequences
        X, y = [], []
        for i in range(len(data)-self.n_steps-self.n_outputs+1):
            X.append(data[i:i+self.n_steps, :])
            y.append(data[i+self.n_steps:i+self.n_steps+self.n_outputs, 0])
        X, y = np.array(X), np.array(y)

        # Split data into training/validation sets
        split_idx_train = int(len(X) * 0.8)
        split_idx_val = int(len(X) * 0.9)

        X_train, y_train = X[:split_idx_train], y[:split_idx_train]
        X_val, y_val = X[split_idx_train:split_idx_val], y[split_idx_train:split_idx_val]
        X_test, y_test = X[split_idx_val:], y[split_idx_val:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_model(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
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
            tf.keras.layers.Dense(self.n_outputs,activation="tanh")
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

    def evaluate_model(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        print(Fore.YELLOW, f"Test Loss: {loss}",Style.RESET_ALL)
        print(self.features)

    def predict(self, data):
        #save data for test
        test = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)[-self.n_steps:]
        # Prepare data for prediction
        data = data[self.features]
        data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)
        data_non_close = data[self.non_close_features]
        # data_close = self.scaler1.transform(data_close)
        data_non_close = self.scaler2.transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

        #Predict the future after test data
        if argv[2] == "test":
            # Make prediction on test data
            X_pred = np.array([data[-self.n_steps*2:-self.n_steps, :]])
            y_pred = self.model.predict(X_pred)
            y_pred = y_pred.flatten()
            print(Fore.RED,y_pred,Style.RESET_ALL)
            print(Fore.GREEN,test.flatten(),Style.RESET_ALL)
            # y_pred = self.scaler1.inverse_transform(y_pred)[0]

            #check accuracy of prediction   
            correct = 0
            incorrect = 0  
            for test_val, pred_val in zip(test.flatten(),y_pred):
                if test_val < 0 and pred_val < 0:
                    correct += 1
                elif test_val > 0 and pred_val > 0:
                    correct += 1
                elif test_val < 0 and pred_val > 0:
                    incorrect += 1
                elif test_val > 0 and pred_val < 0:
                    incorrect += 1
            print('=======================================')
            print(Fore.YELLOW, f'MAPE test data: {round(mean_absolute_percentage_error(test.flatten(),y_pred)*100,2)} %',Style.RESET_ALL)
            print(Fore.YELLOW, f'RMSE test data: {round(mean_squared_error(test.flatten(),y_pred,squared=False),10)}',Style.RESET_ALL)
            print('=======================================')
            print(Fore.GREEN,f'correct direction: {correct / (correct + incorrect)}',Style.RESET_ALL,
                Fore.RED,f'incorrect direction: {incorrect / (correct + incorrect)}',Style.RESET_ALL)
            print('=======================================')
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
        else:
            X_pred = np.array([data[-self.n_steps:, :]])
            y_pred = self.model.predict(X_pred)
            y_pred = y_pred.flatten()
            print(Fore.GREEN,f'next {self.n_steps} days for {self.crypt_name}: {y_pred}',Style.RESET_ALL)

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
        # self.RSI_MACD_calc()
        # Prepare data for training
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(self.data)
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        self.evaluate_model(X_test,y_test)
    
        # Make prediction for the next 30 days
        prediction, last_date = self.predict(self.data)
        if argv[2] != "test":
            print(pd.to_datetime(last_date))
            start_date = pd.to_datetime(last_date).date() + timedelta(days=1)
            end_date = start_date + timedelta(days=len(prediction)-1)
            date_range_array = pd.date_range(start=start_date, end=end_date)
            pd.DataFrame({'pred':prediction,'date':date_range_array}).to_csv(f'{self.crypt_name}_pred.csv',index=False)
            self.plot_results()

changePricePredictor(crypt=argv[1],
                    n_features=10, 
                    n_steps=30, 
                    n_outputs=30, 
                    n_epochs=250, 
                    batch_size=256).run_analysis()
