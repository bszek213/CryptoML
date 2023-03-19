#Predict if tomorrows crypto price will increase or decrease based on technical indicators 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import yfinance as yf
import numpy as np

def get_ohlc(crypt):
    crypt_name = crypt + '-USD'
    temp = yf.Ticker(crypt_name)
    data = temp.history(period = 'max', interval="1d")
    save_file = crypt + '.csv'
    data.to_csv(save_file,index=False)
    return data

def OBV(data):
        """
        If the closing price of the asset is higher than the previous day?s closing price: 
        OBV = Previous OBV + Current Day Volume
        
        If the closing price of the asset is the same as the previous day?s closing price:
        OBV = Previous OBV (+ 0)
        If the closing price of the asset is lower than the previous day?s closing price:
        OBV = Previous OBV - Current Day's Volume
        """
        #TODO: add a linear regression lineover like 10-20 data points and use that 
        # as a buy signal if it is positive
        data['OBV'] = np.zeros(len(data))
        # crypto_df_final['OBV'].iloc[0] = crypto_df_final['volume'].iloc[0]
        OBV_iter = data['Volume'].iloc[0]
        data['OBV'].iloc[0] = OBV_iter
        for i in range(1,len(data['OBV'])):
            if (data['Close'].iloc[i-1] < data['Close'].iloc[i]):
                OBV_iter += data['Volume'].iloc[i]
            if (data['Close'].iloc[i-1] > data['Close'].iloc[i]):
                OBV_iter -= data['Volume'].iloc[i]
            if (data['Close'].iloc[i-1] == data['Close'].iloc[i]):
                OBV_iter += 0
            data['OBV'].iloc[i] = OBV_iter
        return data

def RSI(data):
    update = pd.DataFrame()
    update['change'] = data.Close.diff()
    # crypto_df['U'] = [x if x > 0 else 0 for x in crypto_df.change]
    # crypto_df['D'] = [abs(x) if x < 0 else 0 for x in crypto_df.change]
    update['U']  = update['change'].clip(lower=0)
    update['D'] = -1*update['change'].clip(upper=0)
    update['U'] = update['U'].ewm(span=14,
                min_periods=1).mean()
    update['D'] = update['D'].ewm(span=14,
                min_periods=1).mean()
    update['RS'] = update['U'] / update['D']
    data['RSI'] = 100 - (100/(1+update['RS']))
    return data

def vol_RSI(data):
    update = pd.DataFrame()
    update['change_vol'] = data.Volume.diff()
    # crypto_df['U'] = [x if x > 0 else 0 for x in crypto_df.change]
    # crypto_df['D'] = [abs(x) if x < 0 else 0 for x in crypto_df.change]
    update['U_vol']  = update['change_vol'].clip(lower=0)
    update['D_vol'] = -1*update['change_vol'].clip(upper=0)
    update['U_vol'] = update.U_vol.ewm(span=14,
                min_periods=1).mean()
    update['D'] = update.D_vol.ewm(span=14,
                min_periods=1).mean()
    update['RS_vol'] = update.U_vol / update.D_vol
    data['RSI_vol'] = 100 - (100/(1+update.RS_vol))
    return data

def moving_averages(data):
    data['ewmshort'] = data['Close'].ewm(span=20, min_periods=1).mean() #used to be 50
    data['ewmmedium'] = data['Close'].ewm(span=100, min_periods=1).mean()
    data['ewmlong'] = data['Close'].ewm(span=200, min_periods=1).mean()
    return data

def money_flow_index(data):
    period = 14
    typical_price = (data['Close'] + data['High'] + data['Low']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
            
        elif typical_price[i] < typical_price[i-1]:
            negative_flow.append(money_flow[i-1])
            positive_flow.append(0)
            
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    positive_mf = []
    negative_mf = []
    for i in range(period-1, len(positive_flow)):
        positive_mf.append( sum(positive_flow[i + 1- period : i+1]))
        
    for i in range(period-1, len(negative_flow)):
        negative_mf.append( sum(negative_flow[i + 1- period : i+1]))
    data['MFI'] = np.full([len(typical_price), 1], np.nan)
    temp = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf) ))
    diff_length = len(data['MFI']) - len(temp)
    for inst in temp:
        data['MFI'].iloc[diff_length] = inst
        diff_length += 1
    return data

def macd(data):
    sma_1 = data['Close'].ewm(span=26,
                        min_periods=1).mean()
    sma_2 = data['Close'].ewm(span=12,
                        min_periods=1).mean()
    data['macd_diff'] = sma_2 - sma_1
    data['signal_line'] = data['macd_diff'].ewm(span=9,min_periods=1).mean()
    return data

def split_x_y(df):
    X = df[['OBV', 'macd_diff', 'signal_line','MFI','ewmshort','ewmmedium','ewmlong',
            'RSI_vol','RSI']].fillna(method='bfill').values[:-1]
    #Make sure this is right
    Y = np.where(df['close'].shift(-1).values[:-1] > df['close'].values[:-1], 1, 0)
def main():
    df = get_ohlc('BTC')
    df = OBV(df)
    df = RSI(df)
    df = vol_RSI(df)
    df = moving_averages(df)
    df = money_flow_index(df)
    df = macd(df)
    split_x_y(df)
if __name__ == "__main__":
    main()