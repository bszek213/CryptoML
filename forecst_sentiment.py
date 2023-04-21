# Import libraries
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
import yfinance as yf
from sys import argv
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

# Create and fit the Prophet model
model = Prophet()
model.fit(df)

# Perform cross-validation to tune the model parameters
# cutoffs = pd.date_range(start='2021-03-01', end='2021-03-21', freq='3D')
df_cv = cross_validation(model, initial='180 days', period='3 days', horizon='7 days')

# Get the RMSE of the fit to the data
df_p = performance_metrics(df_cv)
print(df_p.iloc[0])
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

