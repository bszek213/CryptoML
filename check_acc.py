#check how well the test data fits to the predicted data
import pandas as pd
import yfinance as yf
from sys import argv
import matplotlib.pyplot as plt

pred = pd.read_csv(f'{argv[1]}_pred.csv')
crypt_name = argv[1] + '-USD'
temp = yf.Ticker(crypt_name)
temp = temp.history(period = 'max', interval="1d")
actual =  pd.DataFrame()
actual['Close'] = temp['Close'].pct_change().fillna(method='bfill')
pred['date'] = pd.to_datetime(pred['date'])
actual['date'] = pd.to_datetime(temp.index.values)
merged_df = pd.merge(actual, pred, on='date')
print(merged_df)

plt.scatter(pred['date'],pred['pred'], label='Predicted')
plt.scatter(merged_df['date'],merged_df['Close'], label='Actual')
plt.title(f'{argv[1]} Actual and Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.xticks(rotation=90)
plt.legend()
plt.show()