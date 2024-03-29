#check how well the test data fits to the predicted data
import pandas as pd
import yfinance as yf
from sys import argv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

pred = pd.read_csv(f'{argv[1]}_pred.csv')
crypt_name = argv[1] + '-USD'
temp = yf.Ticker(crypt_name)
temp = temp.history(period = 'max', interval="1d")
actual =  pd.DataFrame()
actual['Close'] = temp['Close'].values#.pct_change().fillna(method='bfill')
pred['date'] = pd.to_datetime(pred['date'])
actual['date'] = pd.to_datetime(temp.index.values)
merged_df = pd.merge(actual, pred, on='date')
print(merged_df)
mape_val = mean_absolute_percentage_error(merged_df["Close"],merged_df["pred"])*100
print(f'MAPE: {mape_val}%')

#Check positive and negative
correct = 0
incorrect = 0
for i in range(len(merged_df)):
    if ((merged_df['Close'].iloc[i] < 0) and 
        (merged_df['pred'].iloc[i] < 0)):
        correct += 1
    elif ((merged_df['Close'].iloc[i] > 0) and 
        (merged_df['pred'].iloc[i] > 0)):
        correct += 1
    elif ((merged_df['Close'].iloc[i] > 0) and 
        (merged_df['pred'].iloc[i] < 0)):
        incorrect += 1
    elif ((merged_df['Close'].iloc[i] < 0) and 
        (merged_df['pred'].iloc[i] > 0)):
        incorrect += 1

print(f'Percent correct in predicting positive and negative days: {(correct/len(merged_df))*100}%')
print(f'Percent incorrect in predicting positive and negative days: {(incorrect/len(merged_df))*100}%')

#plot
plt.figure(figsize=(8,8))
plt.plot(pred['date'],pred['pred'],marker='*',linewidth=1,label='Predicted')
plt.plot(merged_df['date'],merged_df['Close'],marker='*',linewidth=1, label='Actual')
plt.title(f'{argv[1]} Actual and Prediction - current MAPE: {round(mape_val,3)}%')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f'{argv[1]}_accuracy_future_price.png',dpi=300)
plt.show()
