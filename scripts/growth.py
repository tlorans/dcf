import pandas as pd
import numpy as np


risk_free_rate = 0.0425  # Example: 4.25% risk-free rate

income_statements = pd.read_csv('../data/income_statements.csv')
cash_flow_statements = pd.read_csv('../data/cash_flow_statements.csv')

growth_forecast_data = (income_statements
    .get(["symbol", "calendar_year", "net_income", "income_tax_expense", "interest_expense", "interest_income"])
  .assign(
    ebit=lambda x: x["net_income"] + x["income_tax_expense"] + x["interest_expense"] - x["interest_income"]
  )
  .merge(
    (cash_flow_statements
      .rename(columns={
        "change_in_working_capital": "delta_working_capital",
        "capital_expenditure": "capex"
      })
    ), on=["calendar_year", "symbol"], how="inner"
  )
  .assign(
    fcff=lambda x: x["ebit"] + x["depreciation_and_amortization"] - x["income_tax_expense"] + x["delta_working_capital"] - x["capex"]
  )
  .sort_values("calendar_year")
  # keep only symbol, calendar_year and fcff
  .get(["symbol", "calendar_year", "fcff"])
  # drop those with negative or zero FCFF
#   .query("fcff > 0")
  # reset index
  .reset_index(drop=True)
)

# Calculate CAGR over last 5 years for each symbol
def calc_cagr(df):
    df = df.sort_values("calendar_year")
    if len(df) < 5:
        return np.nan  # not enough data
    first_val = df["fcff"].iloc[-5]  # FCFF from 5 years ago
    last_val = df["fcff"].iloc[-1]   # Latest FCFF
    if first_val <= 0 or last_val <= 0:
        return np.nan  # avoid negative or zero in CAGR
    years = df["calendar_year"].iloc[-1] - df["calendar_year"].iloc[-5]
    return (last_val / first_val) ** (1 / years) - 1

cagr_df = (
    growth_forecast_data
    .groupby("symbol", group_keys=False)
    .apply(lambda x: pd.Series({
        "cagr": calc_cagr(x)
    }))
    .reset_index()
    .fillna(0)  # Fill NaN with 0 for symbols with insufficient data
)

# Long-term growth = half of CAGR
cagr_df["long_term_growth"] = cagr_df["cagr"] * 0.5

# if long_term_growth is negative, set it to risk_free_rate
cagr_df["long_term_growth"] = cagr_df["long_term_growth"].clip(lower=0).replace(0, risk_free_rate)

# if long_term_growth is greater than risk_free_rate, set it to risk_free_rate
cagr_df["long_term_growth"] = cagr_df["long_term_growth"].clip(upper=risk_free_rate)


# get the latest FCFF for each symbol
latest_fcff = (
    growth_forecast_data
    .sort_values("calendar_year")
    .groupby("symbol")
    .last()
    .reset_index()[["symbol", "fcff"]]
)
# merge latest FCFF with CAGR data
cagr_df = cagr_df.merge(latest_fcff, on="symbol", how="left")


print(cagr_df)

# save forecast data to CSV
cagr_df.to_csv('../data/growth_forecast.csv', index=False)