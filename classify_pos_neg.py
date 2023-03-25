#Predict if tomorrows crypto price will increase or decrease based on technical indicators 
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM, LeakyReLU, Activation,BatchNormalization#, GRU
from keras.models import Sequential
from keras import optimizers
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.models import Sequential
import yfinance as yf
import numpy as np
# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.ensemble import RandomForestClassifier
# from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sys import argv

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
        positive_mf.append(sum(positive_flow[i + 1- period : i+1]))
        
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

def aroon_ind(data,lb=25):
        """
        AROON UP = [ 25 - PERIODS SINCE 25 PERIOD HIGH ] / 25 * [ 100 ]
        AROON DOWN = [ 25 - PERIODS SINCE 25 PERIOD LOW ] / 25 * [ 100 ]
        if up[i] >= 70 and down[i] <= 30: buy
        if up[i] <= 30 and down[i] >= 70: sell
        """
        data['aroon_up'] = 100 * ((data['High'].rolling(lb).apply(lambda x: x.argmax())) / lb)
        data['aroon_down'] = 100 * ((data['Low'].rolling(lb).apply(lambda x: x.argmin())) / lb)
        return data

def stoch_RSI(data):
        min_val  = data['RSI'].rolling(window=14, center=False).min()
        max_val = data['RSI'].rolling(window=14, center=False).max()
        data['Stoch_RSI'] = ((data['RSI'] - min_val) / (max_val - min_val)) * 100
        return data

def split_x_y(df):
    X = df[['OBV', 'macd_diff', 'signal_line','MFI','ewmshort','ewmmedium','ewmlong',
            'RSI_vol','RSI','aroon_up','Stoch_RSI']].fillna(method='bfill')#.values[:-1]
    #Make sure this is right
    Y = np.where(df['Close'].shift(-1).values > df['Close'].values, 1, 0)
    Y = np.pad(Y, (1, 0), mode='constant')
    Y = Y[:-1]
    df['label'] = Y
    # df[['Close','label']].to_csv('check.csv',index=False)
    # print(df[['Close','label']].head(10))
    # input()
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, train_size=0.80, random_state=42)
    # print(df[['Close','label']])
    # input()
    # # temp = df['label'].value_counts()
    # # print(f'count how many positive and negative days: {temp}')
    # train_size = int(len(X) * 0.80)
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = Y[:train_size], Y[train_size:]
    # print(len(X_train))
    # print(len(y_train))
    # print(len(X_test))
    # print(len(y_test))
    # input()

    return X_train, X_test, y_train, y_test
def deep_model(X_train,y_train,X_test,y_test):
    model = Sequential()
    model.add(Dense(9, input_shape=(X_train.shape[1],)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
    model.summary()
    
    #run this to see the tensorBoard: tensorboard --logdir=./logs
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
    model.fit(X_train,y_train,epochs=3000, batch_size=64, verbose=0,
                         validation_data=(X_test,y_test),callbacks=[tensorboard_callback]) #X_train.reshape(X_train.shape[0], X_train.shape[1], 1
    # model.save('classify_deep.h5')
    model_name = argv[1] + "_model"
    tf.keras.models.save_model(model,model_name)
    # y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    # y_pred_class = np.where(y_pred > 0.5, 1, 0)
    # print(classification_report(y_test, y_pred_class))
def random_forest(df):
    X = df[['OBV', 'macd_diff', 'signal_line','MFI','ewmshort','ewmmedium','ewmlong',
            'RSI_vol','RSI','aroon_up','Stoch_RSI']].fillna(method='bfill')#.values[:-1]
    #Make sure this is right
    Y = np.where(df['Close'].shift(-1).values > df['Close'].values, 1, 0)
    Y = np.pad(Y, (1, 0), mode='constant')
    Y = Y[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    RandForclass = RandomForestClassifier()
    Rand_perm = {
        'criterion' : ["gini","entropy"], #absolute_error - takes forever to run
        'n_estimators': range(300,500,100),
        # 'min_samples_split': np.arange(2, 5, 1, dtype=int),
        'max_features' : [1, 'sqrt', 'log2'],
        'max_depth': np.arange(2,8,1),
        'min_samples_leaf': np.arange(1,3,1)
        }
    clf_rand = GridSearchCV(RandForclass, Rand_perm, 
                        scoring=['accuracy'],
                        cv=5,
                        refit='accuracy',verbose=4, n_jobs=-1)
    search_rand = clf_rand.fit(X_train,y_train)
    joblib_name = "./" + "classifier_" + argv[1] + ".joblib"
    joblib.dump(search_rand, joblib_name, compress=9)
    print('RandomForestClassifier - best params: ',search_rand.best_params_)
    y_pred = search_rand.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

def main():
    #Possible Cryptos: BTC, ETH, DOGE, LTC, SOL
    print(f'creating model for {argv[1]}')
    crypt = argv[1]
    df = get_ohlc(crypt)
    df = OBV(df)
    df = RSI(df)
    df = vol_RSI(df)
    df = moving_averages(df)
    df = money_flow_index(df)
    df = macd(df)
    df = aroon_ind(df)
    df = stoch_RSI(df)
    X_train, X_test, y_train, y_test = split_x_y(df)
    deep_model(X_train,y_train,X_test,y_test)
    random_forest(df)
if __name__ == "__main__":
    main()