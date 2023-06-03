# Cryptocurrency Time Series Forecasting

Deep learning, mainly LSTMs, for cryptocurrency with yfinance

Current MAPE on test data for different cryptocurrencies:
BTC: 127.45% error
ETH: 184.52% error
DOGE: 307.97% error

## Installation
```bash
conda env create -f deep.yaml 
or 
bash -i bash_conda_install.sh
```

## Usage

```python
# Time Series Forecasting - cumulative log returns
python cryoto_deep_many_features.py BTC test
python cryoto_deep_many_features.py BTC future
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Current predictions
![alt text](https://github.com/bszek213/CryptoML/blob/dev/final_prediction.png)
## Training and Validation Loss
![alt text](https://github.com/bszek213/CryptoML/blob/dev/train_val_loss.png)