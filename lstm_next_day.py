#Predict the next day price - regression like

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout #, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sys import argv
import os
from tensorflow.keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
# Define the input and output data
# def create_dataset(dataset, look_back=1):
#     X, y = [], []
#     for i in range(len(dataset) - look_back):
#         X.append(dataset[i:i + look_back, :-1])
#         y.append(dataset[i + look_back, -1])
#     print(X)
#     input()
#     X = np.array(X).reshape(-1, look_back, 1)
#     y = np.array(y)
#     return X, y

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

class nextDay():
    def __init__(self):
        print('instantiate nextDay class object')
        self.crypto = argv[1]
        self.n_steps = 60 #Increasing the sequence length allows the model to capture more long-term dependencies in the data, but can also increase the computational cost of training the model.
    # prepare the data for input into the LSTM model
    def read_data(self):
        # Download BTC data from Yahoo Finance
        name = self.crypto + '-USD'
        self.data = yf.download(name, period='max')
    def get_percent_change(self):
        # Prepare the data
        self.data = self.data[['Close']]
        # self.data['pct_change'] = self.data['Close'].pct_change()
        self.data = self.data.dropna()
    def normalize_data(self):
        # Normalize the data
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(self.data['Close'].values.reshape(-1,1))
        # self.data_scaled = self.scaler.fit_transform(self.data['pct_change'].values.reshape(-1,1))
        # self.data_scaled = self.data['pct_change'].to_numpy().reshape(-1,1)
    def split_data(self):
        # Split the data into training and test sets
        # train_size = int(len(self.data) * 0.75)
        # self.train_data = self.data_scaled[:train_size]
        # self.test_data = self.data_scaled[train_size:]
        split_idx = int(len(self.X) * 0.8)
        self.X_train, self.X_test = self.X[:split_idx], self.X[split_idx:]
        self.y_train, self.y_test = self.y[:split_idx], self.y[split_idx:]
        print('==============================================')
        print(f'length of traininig set: {len(self.X_train)}')
        print(f'length of test set: {len(self.X_test)}')
        print('==============================================')
    def create_seq(self):
        # data = self.data['pct_change'].values
        data = self.data_scaled
        self.X, self.y = prepare_data(data, self.n_steps)
    def algo(self):
        name = "LSTM_next_day_model_" + self.crypto + ".h5"
        if os.path.exists(name):
            self.model = load_model(name)
        else:
            # Define the deep neural network model
            self.model = Sequential()
            self.model.add(LSTM(units=16, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=8, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=4, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            # self.model.add(Dropout(0.2))
            # self.model.add(LSTM(units=8, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1, activation='linear')) #I can use this as I only have positive values
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
            self.model.summary()
            # Train the model
            #run this to see the tensorBoard: tensorboard --logdir=./logs
            # tensorboard_callback = TensorBoard(log_dir="./logs")
            # Define the ReduceLROnPlateau callback
            lr_reducer = ReduceLROnPlateau(factor=0.1, patience=10, verbose=1)
            es = EarlyStopping(monitor="val_loss", min_delta=0.005, verbose=1, patience=15, restore_best_weights=True)
            self.model.fit(self.X_train, self.y_train, epochs=100, 
                        validation_data=(self.X_test, self.y_test),
                        batch_size=self.n_steps,verbose=1,
                        callbacks=[lr_reducer,es]
                        )
            self.model.save(name)
        # # Make predictions
        # y_pred = self.model.predict(self.X_test)
        # y_pred_unscaled = self.scaler.inverse_transform(np.concatenate((self.X_test[:, -1, :-1], y_pred.reshape(-1, 1)), axis=1))[:, -1]
        # y_test_unscaled = self.scaler.inverse_transform(np.concatenate((self.X_test[:, -1, :-1], self.y_test.reshape(-1, 1)), axis=1))[:, -1]

        # Print the predicted percent change for the next day
        # print(f'Predicted BTC percentage change for tomorrow: {y_pred_unscaled[-1]:.4f}')
    def predict_future(self):
        # # use the model to predict the next 2 days of prices
        # last_days = self.data['pct_change'].tail(self.n_steps).values.reshape(-1, 1)
        last_days = self.data['Close'].tail(self.n_steps).values.reshape(-1, 1)
        last_days = self.scaler.transform(last_days)
        last_days = last_days.reshape(1, self.n_steps, 1)
        pred = self.model.predict(last_days)
        # extract the predicted price
        self.future_change = self.scaler.inverse_transform(pred[0])
        predicted_price = self.scaler.inverse_transform(pred[0])[0][0]
    
        print('Predicted price for tomorrow: ', predicted_price)
        #Save to file
        tomorrow = datetime.now() + timedelta(days=1)
        dict_save = {'predict_percent':[predicted_price*100],'date':[str(tomorrow)]}
        filename = "next_day_"+self.crypto+'.csv'
        file_exists = os.path.isfile(filename)
        df = pd.DataFrame(dict_save)
        if file_exists:
            read_df = pd.read_csv(filename)
            concat_df = pd.concat([read_df, df])
            concat_df.to_csv(filename,index=False)
        else:
            df.to_csv(filename, index=False)

    def plot_forecast_pct_change(self):
        today = datetime.now()
        time_array = pd.date_range(start=today, periods=self.n_steps, freq='D')
        list_new_close = []
        print(self.future_change)
        data_start = self.data['Close'].iloc[-1]
        for datum in self.future_change:
            data_start = float(data_start + (data_start * datum))
            list_new_close.append(data_start)
        new_data = pd.DataFrame({
            'Close': list_new_close,
            'pct_change': self.future_change.flatten()
        },index=time_array)
        plt.figure(figsize=(16,8))
        plt.title(f'{argv[1]} Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(self.data.index[-30:], self.data['Close'].iloc[-30:], label='Actual Price',color='tab:blue')
        plt.plot(new_data.index, new_data['Close'], label='Predicted Price',color='tab:orange',marker='*')
        plt.legend()
        plt.show()
    
    def plot_forecast_close(self):
        today = datetime.now()
        time_array = pd.date_range(start=today, periods=self.n_steps, freq='D')
        # print(self.future_change)
        new_data = pd.DataFrame({
            'Close_new': self.future_change.flatten()
        },index=time_array)
        plt.figure(figsize=(16,8))
        plt.title(f'{argv[1]} Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(self.data.index[-30:], self.data['Close'].iloc[-30:], label='Actual Price',color='tab:blue')
        plt.plot(new_data.index, new_data['Close_new'], label='Predicted Price',color='tab:orange',marker='*')
        plt.legend()
        plt.show()

    def plot_next_day(self):
        # get tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        new_row = pd.DataFrame({
            'Close': [0.0],  # replace with actual closing price for tomorrow
            'pct_change': self.future_change[0]  # replace with actual percentage change for tomorrow
        }, index=[tomorrow])
        # append the new row to the DataFrame
        self.data = self.data.append(new_row)
        # sort the DataFrame by the 'Date' index
        self.data.sort_index(inplace=True)
        plt.figure(figsize=(16,8))
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Percent change')
        split_idx = int(len(self.X) * 0.999)
        plt.plot(self.data.index[split_idx:], self.data['pct_change'].iloc[split_idx:], label='Actual Price')
        plt.plot(self.data.index[-2:], self.data['pct_change'].iloc[-2:],marker='*', color='tab:orange',label='Predicted Price')
        plt.legend()
        plt.show()
    def run_analysis(self):
        self.read_data()
        self.get_percent_change()
        self.normalize_data()
        self.create_seq()
        self.split_data()
        self.algo()
        self.predict_future()
        self.plot_forecast_close()
def main():
    nextDay().run_analysis()
if __name__ == "__main__":
    main()
