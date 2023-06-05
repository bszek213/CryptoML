# Cryptocurrency Time Series Forecasting

Deep learning, mainly LSTMs, for cryptocurrency with yfinance

Current MAPE on test data for different cryptocurrencies:
BTC: 2.21% error
ETH: 184.52% error
DOGE: 245.15% error
TRX: 108.81% error
MANA: 171.94% error
LTC: 114.19% error
DOT: 105.18% error
BCH: 136.55% error

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