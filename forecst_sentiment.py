# Import libraries
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
import yfinance as yf
from sys import argv
import os
import pickle
from prophet.serialize import model_to_json, model_from_json

def tune_params(df):
    """
    bitcoin best_params: {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10, 'holidays_prior_scale': 0.01}
    ethereum best_params: {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10, 'holidays_prior_scale': 0.01}: rmse,mape [14.520390078172557, 0.3444106823120257]
    flow best_params: {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.01, 'holidays_prior_scale': 0.01}: rmse,mape [13.463912485965324, 0.3593867745229026]
    """
    # Define parameter grid to try out
    params = {
        'changepoint_prior_scale': [0.01, 0.1, .5],
        'seasonality_prior_scale': [0.01, 1, 10],
        'holidays_prior_scale': [0.01, 1, 10],
    }

    # Initialize best parameters and error values
    best_params = None
    best_rmse = float('inf')
    best_mape = float('inf')

    # Loop over all combinations of parameters
    for changepoint_prior_scale in params['changepoint_prior_scale']:
        for seasonality_prior_scale in params['seasonality_prior_scale']:
            for holidays_prior_scale in params['holidays_prior_scale']:
                # Create and fit the Prophet model with current parameter combination
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    interval_width=0.95
                )
                model.fit(df)
                
                # Perform cross-validation and compute performance metrics
                df_cv = cross_validation(model, initial='180 hours', period='1 hours',horizon='30 minutes')
                df_p = performance_metrics(df_cv)
                rmse = df_p['rmse'].values[0]
                mape = df_p['mape'].values[0]
                
                # Check if current model has lower RMSE and MAPE than previous best model
                if rmse < best_rmse and mape < best_mape:
                    best_params = {
                        'changepoint_prior_scale': changepoint_prior_scale,
                        'seasonality_prior_scale': seasonality_prior_scale,
                        'holidays_prior_scale': holidays_prior_scale
                    }
                    best_rmse = rmse
                    best_mape = mape
    print(f'{argv[2]} best_params: {best_params}: rmse,mape {[best_rmse,best_mape]}')
    return model
# Load the data
#Sentiment
if argv[1] == 'sentiment':
    data = pd.read_csv("fear_greed.csv")
    data['date'] = pd.to_datetime(data['date'])
    # Create the Prophet dataframe
    df = data.rename(columns={'date': 'ds', 'fear_greed': 'y'})

#Crypto
if argv[1] == 'crypto':
    name = argv[2] + '-USD'
    data = yf.download(name, period='max')
    data['date'] = pd.to_datetime(data.index.values)
    df = data.rename(columns={'date': 'ds', 'Close': 'y'})
    df = df[['ds','y']]

if argv[1] == "all":
    name = argv[2]
    loc_1 = "/home/brianszekely/Desktop/ProjectsResearch/technical-analysis-crypto/fear_greed_all_cryptos.csv"
    loc_2 = "/home/bszekely/Desktop/crypto_short_update/fear_greed_all_cryptos.csv"
    data = pd.read_csv(loc_1)
    print('=====================')
    print(f'Data Length: {len(data)}')
    print('=====================')
    #data['date'] = pd.datetime(data['timestamp'])
    df = data.rename(columns={'timestamp': 'ds', name : 'y'})
    df = df[['ds','y']]
    #df['ds'] = df['ds'].str.split().str[0]
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna()

# Create and fit the Prophet model: parameter tune 
# Perform cross-validation to tune the model parameters
if argv[1] == "all":
    if os.path.exists(f'{argv[2]}_prophet_model.pickle'):
        # Load the model from disk
        with open(f'{argv[2]}_prophet_model.pickle', 'rb') as f:
            model = model_from_json(pickle.load(f))
        model.fit(df)
    else:
        model = tune_params(df)
        with open(f'{argv[2]}_prophet_model.pickle', 'wb') as f:
            pickle.dump(model_to_json(model), f)
    # model.fit(df)
    # df_cv = cross_validation(model, initial='180 hours', period='1 hours',horizon='30 minutes')
    print('Model is tuned')
else:
    model = Prophet(interval_width=0.95)
    model.fit(df)
    df_cv = cross_validation(model, initial='180 days', period='3 days', horizon='7 days')
    # Get the RMSE of the fit to the data
    df_p = performance_metrics(df_cv)
    print('=====================')
    print(df_p.iloc[0])
    print('=====================')

if argv[1] == "all":
    # Predict the next 7 days
    future = model.make_future_dataframe(periods=1*24*4, freq='15T') # 1 days, 15 minutes interval
    forecast = model.predict(future)
    predicted_values = forecast['yhat'][-1*24*4:] # last 1 days, 15 minutes interval
    #find closest date 
    last_date = df['ds'].iloc[-1]
    closest_idx = (forecast['ds'] - last_date).abs().idxmin()
    if forecast['ds'].loc[closest_idx] < last_date:
        # closest date is earlier than last date in df1
        closest_idx += 1
    closest_row = forecast.loc[closest_idx]

    if df['y'].iloc[-1] < closest_row['yhat_lower']:
        print('buy')
    else:
        print('no buy')
else:
    # Predict the next 7 days
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    predicted_values = forecast['yhat'][-7:]

if argv[2]:
    title_name = argv[2]
else:
    title_name = "sentiment"
fig = plot_plotly(model, forecast)
fig.update_layout(title=title_name, width=1300, height=1000)
fig.show()

fig_components = plot_components_plotly(model, forecast)
fig_components.update_layout(title=title_name, width=1300, height=1000)
fig_components.show()

