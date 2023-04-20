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
        self.data['pct_change'] = self.data['Close'].pct_change()
        self.data = self.data.dropna()
    def normalize_data(self):
        # Normalize the data
        self.scaler = MinMaxScaler()
        self.data_scaled = self.scaler.fit_transform(self.data['pct_change'].values.reshape(-1,1))
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
        name = "LSTM_next_day_model" + self.crypto + ".h5"
        if os.path.exists(name):
            self.model = load_model(name)
        else:
            # Define the deep neural network model
            self.model = Sequential()
            self.model.add(LSTM(units=32, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=16, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=8, activation='relu', return_sequences=True, input_shape=(self.n_steps, 1)))
            # self.model.add(Dropout(0.2))
            # self.model.add(LSTM(units=8, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
            self.model.summary()
            # Train the model
            #run this to see the tensorBoard: tensorboard --logdir=./logs
            tensorboard_callback = TensorBoard(log_dir="./logs")
            self.model.fit(self.X_train, self.y_train, epochs=50, 
                        validation_data=(self.X_test, self.y_test),
                        batch_size=self.n_steps,verbose=1,
                        callbacks=[tensorboard_callback]
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
        last_days = self.data['pct_change'].tail(self.n_steps).values.reshape(-1, 1)
        last_days = self.scaler.transform(last_days)
        last_days = last_days.reshape(1, self.n_steps, 1)
        pred = self.model.predict(last_days)
        # extract the predicted price
        self.future_change = self.scaler.inverse_transform(pred[0])
        predicted_price = self.scaler.inverse_transform(pred[0])[0][0]
        print('Predicted price for tomorrow: ', predicted_price)
        #Save to file
        dict_save = {'Tomorrow_predict_percent':[predicted_price*100],'date':[str(datetime.now())]}
        filename = "next_day_"+self.crypto+'.csv'
        file_exists = os.path.isfile(filename)
        df = pd.DataFrame(dict_save)
        if file_exists:
            read_df = pd.read_csv(filename)
            concat_df = pd.concat([read_df, df])
            concat_df.to_csv(filename,index=False)
        else:
            df.to_csv(filename, index=False)

    def plot_output(self):
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
        self.plot_output()
def main():
    nextDay().run_analysis()
if __name__ == "__main__":
    main()
