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

prices_daily = tf.download_data(
  domain="stock_prices", 
  symbols=symbols["symbol"].tolist(),
  start_date="2000-01-01", 
  end_date="2023-12-31"
)


prices_daily = (prices_daily
  .groupby("symbol")
  .apply(lambda x: x.assign(counts=x["adjusted_close"].dropna().count()))
  .reset_index(drop=True)
  .query("counts == counts.max()")
)

returns_monthly = (prices_daily
  .assign(
    date=prices_daily["date"].dt.to_period("M").dt.to_timestamp()
  )
  .groupby(["symbol", "date"], as_index=False)
  .agg(adjusted_close=("adjusted_close", "last"))
  .assign(
    ret=lambda x: x.groupby("symbol")["adjusted_close"].pct_change()
  )
)

import pandas_datareader as pdr

factors_raw = pdr.DataReader(
  name="F-F_Research_Data_5_Factors_2x3",
  data_source="famafrench", 
  start="2020-01-01", 
  end="2025-08-01")[0]

factors = (factors_raw
  .divide(100)
  .reset_index(names="date")
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
  .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)

import statsmodels.formula.api as smf

returns_excess_monthly = (returns_monthly
  .merge(factors, on="date", how="left")
  .assign(ret_excess=lambda x: x["ret"] - x["rf"])
)

def estimate_capm(data):
  model = smf.ols("ret_excess ~ mkt_excess", data=data).fit()
  result = pd.DataFrame({
    "coefficient": ["alpha", "beta"],
    "estimate": model.params.values,
    "t_statistic": model.tvalues.values
  })
  return result

capm_results = (returns_excess_monthly
  .groupby("symbol", group_keys=True)
  .apply(estimate_capm)
  .reset_index()
)

alphas = (capm_results
  .query("coefficient=='alpha'")
  .assign(is_significant=lambda x: np.abs(x["t_statistic"]) >= 1.96)
)

alphas["symbol"] = pd.Categorical(
  alphas["symbol"],
  categories=alphas.sort_values("estimate")["symbol"],
  ordered=True
)

alphas_figure = (
  ggplot(
    alphas, 
    aes(y="estimate", x="symbol", fill="is_significant")
  )
  + geom_col()
  + scale_y_continuous(labels=percent_format())
  + coord_flip()
  + labs(
      x="Estimated asset alphas", y="", fill="Significant at 95%?",
      title="Estimated CAPM alphas for Dow index constituents"
    )
)
alphas_figure.save("../images/alphas.png", dpi=300)



betas = (capm_results
  .query("coefficient=='beta'")
  .assign(is_significant=lambda x: np.abs(x["t_statistic"]) >= 1.96)
)

betas["symbol"] = pd.Categorical(
  betas["symbol"],
  categories=betas.sort_values("estimate")["symbol"],
  ordered=True
)

betas_figure = (
  ggplot(
    betas,
    aes(y="estimate", x="symbol", fill="is_significant")
  )
  + geom_col()
  + scale_y_continuous()
  + coord_flip()
  + labs(
      x="Estimated asset betas", y="", fill="Significant at 95%?",
      title="Estimated CAPM betas for Dow index constituents"
    )
)
betas_figure.save("../images/betas.png", dpi=300)

print(betas)

# save betas to CSV
betas.to_csv("../data/betas.csv", index=False)