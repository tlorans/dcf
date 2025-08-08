import pandas as pd

market_cap = pd.read_csv("../data/market_caps.csv")

# Load income statements
net_debt = (
    pd.read_csv("../data/balance_sheet_statements.csv")
    .filter(items=[
        "symbol", "calendar_year", "net_debt"
    ])
    .assign(fiscal_year=lambda x: x["calendar_year"].astype(int))
    .sort_values(["symbol", "calendar_year"], ascending=[True, False])
    .groupby("symbol", as_index=False)
    .head(1)
    .filter(items=[
        "symbol", "calendar_year", "net_debt",
    ])
)

# Merge market cap with net debt
capital_structure = (market_cap.merge(
    net_debt, on=["symbol"], how="inner"
)
    .assign(
        total_assets=lambda x: x["market_cap"] + x["net_debt"],
    )
    .assign(
        w_equity=lambda x: x["market_cap"] / (x["total_assets"]),
        w_debt=lambda x: x["net_debt"] / (x["total_assets"]),
    )
    .filter(items=[
        "symbol", "calendar_year", "w_equity", "w_debt"
    ])
)

print(capital_structure)

# Save the capital structure to a CSV file
capital_structure.to_csv("../data/capital_structure.csv", index=False)