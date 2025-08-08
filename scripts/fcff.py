import pandas as pd
import numpy as np
from plotnine import *
from mizani.formatters import percent_format, comma_format
from itertools import product

income_statements = pd.read_csv('../data/income_statements.csv')
cash_flow_statements = pd.read_csv('../data/cash_flow_statements.csv')

dcf_data = (income_statements
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
  .reset_index(drop=True)
)

print(dcf_data)