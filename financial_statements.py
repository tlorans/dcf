import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fmpapi import fmp_get
from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text

load_dotenv()

print(fmp_get(
  resource="balance-sheet-statement", 
  symbol="MSFT", 
  params={"period": "annual", "limit": 5},
  to_pandas=True
))