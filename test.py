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

print(symbols)
print(len(symbols))