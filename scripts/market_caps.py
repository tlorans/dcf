import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import tidyfinance as tf

from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text

symbols = tf.download_data(
  domain="constituents", 
  index="Dow Jones Industrial Average"
)

# Get market cap for all symbols in your list
tickers = symbols["symbol"].tolist()
market_caps = []

for ticker in tickers:
    info = yf.Ticker(ticker).fast_info
    market_caps.append({"symbol": ticker, "market_cap": info.get("marketCap"), "shares_outstanding": info.get("shares")})

market_caps_df = pd.DataFrame(market_caps)
print(market_caps_df)

# Save the market caps to a CSV file
market_caps_df.to_csv("../data/market_caps.csv", index=False)
