import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import tidyfinance as tf
from dotenv import load_dotenv
from fmpapi import fmp_get
from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text

symbols = tf.download_data(
  domain="constituents", 
  index="Dow Jones Industrial Average"
)

list_symbols = symbols["symbol"].tolist()

load_dotenv()

params = {"period": "annual", "limit": 5}

balance_sheet_statements = pd.concat(
  [fmp_get(
      resource="balance-sheet-statement", symbol=x, params=params, to_pandas=True
    ) for x in list_symbols],
  ignore_index=True
)

income_statements = pd.concat(
  [fmp_get(
      resource="income-statement", symbol=x, params=params, to_pandas=True
    ) for x in list_symbols],
  ignore_index=True
)

cash_flow_statements = pd.concat(
  [fmp_get(
      resource="cash-flow-statement", symbol=x, params=params, to_pandas=True
    ) for x in list_symbols],
  ignore_index=True
)

# Save the data to CSV files
balance_sheet_statements.to_csv("../data/balance_sheet_statements.csv", index=False)
income_statements.to_csv("../data/income_statements.csv", index=False)
cash_flow_statements.to_csv("../data/cash_flow_statements.csv", index=False)