from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text
import pandas as pd
import numpy as np
import tidyfinance as tf

# Load income statements
income_statements = (
    pd.read_csv("../data/income_statements.csv")
    .filter(items=[
        "symbol", "calendar_year", "interest_expense",
        "income_before_tax", "income_tax_expense"
    ])
    .assign(fiscal_year=lambda x: x["calendar_year"].astype(int))
)

# Load balance sheet statements
balance_sheet_statements = (
    pd.read_csv("../data/balance_sheet_statements.csv")
    .filter(items=["symbol", "calendar_year", "total_debt"])
    .assign(fiscal_year=lambda x: x["calendar_year"].astype(int))
)

# Merge datasets
financial_statements = pd.merge(
    income_statements,
    balance_sheet_statements,
    on=["symbol", "calendar_year"]
)

# Effective tax rate per year (avoid div/0)
financial_statements = financial_statements.assign(
    effective_tax_rate=lambda x: np.where(
        x["income_before_tax"] != 0,
        x["income_tax_expense"] / x["income_before_tax"],
        np.nan
    )
)

# Average effective tax rate per company
avg_tax_rate_per_symbol = (
    financial_statements
    .groupby("symbol")["effective_tax_rate"]
    .mean()
    .reset_index(name="avg_effective_tax_rate")
)

# Merge back into main DataFrame
financial_statements = financial_statements.merge(
    avg_tax_rate_per_symbol, on="symbol", how="left"
)

# Compute cost of debt and after-tax cost of debt
financial_statements = (
    financial_statements
    .assign(
        cost_of_debt=lambda x: x["interest_expense"] / x["total_debt"],
        after_tax_cost_of_debt=lambda x: x["cost_of_debt"] * (1 - x["avg_effective_tax_rate"])
    )
    .filter(items=[
        "symbol", "calendar_year", "cost_of_debt",
        "avg_effective_tax_rate", "after_tax_cost_of_debt"
    ])
)

# Keep only latest year per symbol
financial_statements = (
    financial_statements
    .sort_values(["symbol", "calendar_year"], ascending=[True, False])
    .groupby("symbol", as_index=False)
    .head(1)
    .filter(items=[
        "symbol", "calendar_year", "cost_of_debt",
        "avg_effective_tax_rate", "after_tax_cost_of_debt"
    ])
)

print(financial_statements.head())

# now get the latest year for each symbol

# Save the cost of debt data to CSV
financial_statements.to_csv("../data/cost_of_debt.csv", index=False)