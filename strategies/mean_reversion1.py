from gemini import engine, helpers
import pandas as pd
from mercury.models import Candle
from datetime import datetime, timezone
from typing import List
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Algorithm
def logic(account, lookback):
    try:
        lookback = helpers.period(lookback)
        today = lookback.loc(0)

    except Exception as e:
        print(e)
        pass # Handles lookback errors in beginning of dataset


# Setup
candles = {}
symbols = ['ETH', 'ZEC', 'XRP', 'LTC']
LOOKBACK = 512
start_date = datetime(2017, 12, 20)
end_date = datetime(2018, 6, 20)

def get_candles(symbols: List[str], start_date: datetime, end_date: datetime):
    return Candle.select(Candle.timestamp, Candle.symbol, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume).where(
      Candle.timestamp >= start_date.replace(tzinfo=timezone.utc).timestamp(),
      Candle.timestamp < end_date.replace(tzinfo=timezone.utc).timestamp(),
      Candle.symbol.in_(symbols)).order_by(
        Candle.timestamp.asc()
      )

def joh_output(res):
    output = pd.DataFrame([res.lr2, res.lr1],
                          index=['max_eig_stat', "trace_stat"])

    print(output.T, '\n')
    print("Critical values(90%, 95%, 99%) of max_eig_stat\n", res.cvm, '\n')
    print("Critical values(90%, 95%, 99%) of trace_stat\n", res.cvt, '\n')
    print("Eigenvalues of VECM coef matrix\n", res.eig, "\n")
    print("Eigenvectors of VECM coef matrix\n", res.evec, "\n")
    print()

# Execution
candles = get_candles(symbols, start_date, end_date)

# Dataframe
df_all = pd.DataFrame()
df_all = pd.DataFrame(list(candles.dicts()))
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], unit='s')
df = df_all.pivot(index='timestamp', columns='symbol', values='close')
df = df[symbols]

# # ADF for each series
# for col in df.columns:
#     # Print the p-value as a percent
#     print(f'{col} - p-value:', round(adfuller(df[col])[1] * 100, 2), '%')

# Johansen
johansen_res = coint_johansen(df, 0, 1)
# joh_output(johansen_res)

# # Transposed eigenvector matrix
df_eigvec_t = pd.DataFrame(johansen_res.evec).T

# # Unit portfolio timeseries
cols_ohlcv = ['open', 'high', 'low', 'close', 'volume']
cols = df.columns
ev = df_eigvec_t.iloc[0]
df_port = pd.DataFrame()
for col_ohlcv in cols_ohlcv:
    df_temp = df_all.pivot(index='timestamp', columns='symbol', values=col_ohlcv)
    df_port[col_ohlcv] = sum(ev[i] * df_temp[col] for i, col in enumerate(cols))

adf_port = adfuller(df_port['close'], maxlag=1, regression='c', autolag=None)

# Create indicators for df_port
df_port['MA'] = df_port['close'].rolling(window=LOOKBACK).mean()
MSTD = df_port['close'].rolling(window=LOOKBACK).std()
df_port['lower'] = df_port['MA'] - 2 * MSTD
df_port['upper'] = df_port['MA'] + 2 * MSTD
Z_SCORE = (df_port['close'] - df_port['MA']) / MSTD

# # Figure - Time series
# plt.style.use('dark_background')
# fig, axes = plt.subplots(4, 1, figsize=(18, 12))
# axes[0].plot(df)
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Price')
# axes[0].set(xlim=(df.index.values[0], df.index.values[-1]))
# axes[0].legend(df.columns)

# # Figure - Normalized Time series
# axes[1].plot(df.divide(df.iloc[0]))
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Price')
# axes[1].set(xlim=(df.index.values[0], df.index.values[-1]))
# axes[1].legend(df.columns)

# # Figure - Portfolio + bbands
# axes[2].plot(df_port['close'])
# axes[2].set_xlabel('Time')
# axes[2].set_ylabel('Unit Portfolio Price')
# axes[2].set(xlim=(df.index.values[0], df.index.values[-1]))
# axes[2].legend(df_port.columns)

# # Figure - Z-score
# axes[3].plot(Z_SCORE)
# axes[3].set_xlabel('Time')
# axes[3].set_ylabel('Z-score')
# axes[3].axhline(0, color='red', linestyle='--')
# axes[3].axhline(1, color='green', linestyle='-.')
# axes[3].axhline(-1, color='green', linestyle='-.')
# axes[3].axhline(2, color='yellow', linestyle='dotted')
# axes[3].axhline(-2, color='yellow', linestyle='dotted')
# axes[3].set(xlim=(df.index.values[0], df.index.values[-1]))
# axes[3].legend()

# plt.show()

# Backtest & Massage data into what's expected by Gemini
df_port.index.set_names('date', inplace=True)
df_port = df_port.reset_index()
backtest = engine.backtest(df_port)
backtest.start(1000, logic)
backtest.results()
