import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
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
        crypt_name = crypt + '-USD'
        temp = yf.Ticker(crypt_name)
        price_data = temp.history(period = 'max', interval="1d")
        print(Fore.GREEN,f'NUMBER OF SAMPLES FOR {crypt_name}: {len(price_data)}',Style.RESET_ALL)
        self.features = ['Close','Open', 'High', 'Low','Volume', 'Dividends', 'Stock Splits',
                                    'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
                                    'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
                                    'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                                    'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                                    'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                                    'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                                    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                                    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
                                    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                                    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                                    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
                                    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
                                    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                                    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                                    'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
                                    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                                    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                                    'trend_psar_down', 'trend_psar_up_indicator',
                                    'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
                                    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
                                    'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
                                    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                                    'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
                                    'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
                                    'others_cr']
        self.non_close_features = ['Open', 'High', 'Low','Volume', 'Dividends', 'Stock Splits',
                                    'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
                                    'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
                                    'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                                    'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                                    'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                                    'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                                    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                                    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
                                    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                                    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                                    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
                                    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
                                    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                                    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                                    'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
                                    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                                    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                                    'trend_psar_down', 'trend_psar_up_indicator',
                                    'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
                                    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
                                    'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
                                    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                                    'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
                                    'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
                                    'others_cr']
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
        print(self.data.columns)

    def prepare_data(self, data):
        # Extract relevant features
        data = self.data[self.features]

        # Scale data
        self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        # self.scaler2 = StandardScaler()
        # self.scaler1 = StandardScaler()

        # data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1) #pct_change
        
        #Close price
        data_close = data['Close'].to_numpy().reshape(-1, 1) #close price
        data_close = self.scaler1.fit_transform(data_close)

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
            # staircase=True
        )
        drop_val = 0.3
        model = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, activation='relu',return_sequences=True, input_shape=(self.n_steps, self.n_features))),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(drop_val),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5, activation='relu',return_sequences=False)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(7, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5, activation='relu',return_sequences=True,)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(drop_val),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, activation='relu')),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_outputs,activation="relu")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),loss='mean_squared_error')
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
                      shuffle=False,
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
        # test = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)[-self.n_steps:] #pct_change
        test = data['Close'].to_numpy().reshape(-1, 1)[-self.n_steps:] #close
        # Prepare data for prediction
        data = data[self.features]
        # data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1) #pct_change
        data_close = data['Close'].to_numpy().reshape(-1, 1) #close
        data_close = self.scaler1.transform(data_close)
        data_non_close = data[self.non_close_features]
        data_non_close = self.scaler2.transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

        #Predict the future after test data
        if argv[2] == "test":
            # Make prediction on test data
            X_pred = np.array([data[-self.n_steps*2:-self.n_steps, :]])
            #pct-change
            # y_pred = self.model.predict(X_pred)
            # y_pred = y_pred.flatten()
            #close
            y_pred = self.model.predict(X_pred)
            y_pred = self.scaler1.inverse_transform(y_pred)[0]

            print(Fore.RED,y_pred,Style.RESET_ALL)
            print(Fore.GREEN,test.flatten(),Style.RESET_ALL)

            #check accuracy of prediction   
            correct = 0
            incorrect = 0  
            test_pct = np.diff(test.flatten())
            y_pred_pct = np.diff(y_pred)
            for test_val, pred_val in zip(test_pct,y_pred_pct):
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
            print(Fore.YELLOW, "R2 score test data:", r2_score(test.flatten(),y_pred),Style.RESET_ALL)
            print('=======================================')
            print(Fore.GREEN,f'correct direction: {correct / (correct + incorrect)}',Style.RESET_ALL,
                Fore.RED,f'incorrect direction: {incorrect / (correct + incorrect)}',Style.RESET_ALL)
            print('=======================================')
            plt.plot(y_pred,color='r',marker='*',alpha=0.3,label='pred')
            plt.plot(test.flatten(),marker='*',color='g',alpha=0.3,label='test')
            plt.legend()
            plt.show()
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
            y_pred = self.scaler1.inverse_transform(y_pred)[0]
            y_pred = y_pred.flatten()
            print(Fore.GREEN,f'next {self.n_steps} days for {self.crypt_name}: {y_pred}',Style.RESET_ALL)

        return y_pred, self.data.index[-1]
    
    def plot_results(self):
        pred = pd.read_csv(f'{self.crypt_name}_pred.csv')
        # plt.plot(self.data['Close'], label='Actual')
        plt.plot(pred['date'],pred['pred'], marker='*',label='Predicted')
        plt.title(f'{self.crypt_name} Close Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.xticks(rotation=45)
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
