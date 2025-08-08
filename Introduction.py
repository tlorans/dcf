import streamlit as st
import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import tidyfinance as tf

from plotnine import *
from mizani.formatters import percent_format
from adjustText import adjust_text

st.title("Free Cash Flow to the Firm (FCFF) Valuation Model")

st.write(r"""

We think about the firm as a "cash processor". 
         
Cash flows into the firm in the form of revenue as it sells it product, and cash flows out as it pays its cashs operating expenses (salaries and taxes, 
         but not interest expense, which is a financing and not an operating expense). The firm takes the cash that is left over and makes 
         short-term net investments in working capital (e.g., invetory and receivables) and long-term investments in property, plant and equipment. 

After these actions, the cash that remains is available to pay out to the firm's investors: bondholders and common shareholders (assuming no preferred stocks). 
That pile of remaining cash is called free cash flow to the firm (FCFF), because it's "free" to be paid out to the firm's investors.

         """)

st.write(r"""
The formal definition of FCFF is the cash available to all of the firm's investors, including stockholders and bondholders, after the firm buys and sells products,
         provides services, pays its cash operating expenses and makes short-term and long-term investments.
         """)

st.write(r"""
The formula for FCFF is:
         """)

st.latex(r"""
\text{FCFF} = \text{EBIT} + \text{D\&A} - \text{Taxes} + \Delta \text{WC} - \text{CapEx}
         """)

st.write(r"""
where:
- EBIT is Earnings Before Interest and Taxes measures core operating profit 
- D&A is Depreciation and Amortization, which are non-cash expenses that reduce EBIT
- Taxes are the cash taxes paid by the firm
- $\Delta$ WC is the change in working capital, which is the difference between current assets and current liabilities (excluding cash and short-term debt)
- CapEx is Capital Expenditures, which are the firm's investments in long-term assets like property, plant, and equipment
         """)

income_statements = pd.read_csv('./data/income_statements.csv')
cash_flow_statements = pd.read_csv('./data/cash_flow_statements.csv')

# select a symbol from the dropdown
selected_symbol = st.sidebar.selectbox(
    "Select a symbol to view its FCFF data",
    income_statements["symbol"].unique()
)

income_statements_selected = income_statements[income_statements["symbol"] == selected_symbol]
cash_flow_statements_selected = cash_flow_statements[cash_flow_statements["symbol"] == selected_symbol]

# get the latest data for the selected symbol
latest_income_statement = income_statements_selected.sort_values("calendar_year").iloc[-1]
latest_cash_flow_statement = cash_flow_statements_selected.sort_values("calendar_year").iloc[-1]


net_income = latest_income_statement["net_income"]
income_tax_expense = latest_income_statement["income_tax_expense"]
interest_expense = latest_income_statement["interest_expense"]
interest_income = latest_income_statement["interest_income"]
ebit = net_income + income_tax_expense + interest_expense - interest_income

st.write(f"""The EBIT can be computed as:""")

st.latex(r"""
\text{EBIT} = \text{Net Income} + \text{Income Tax Expense} + \text{Interest Expense} - \text{Interest Income}
         """)

# string interpolated formula for EBIT
ebit_formula = f"""
{ebit:,.0f} = {net_income:,.0f} + {income_tax_expense:,.0f} + {interest_expense:,.0f} - {interest_income:,.0f}
"""
st.latex(ebit_formula)

depreciation_and_amortization = latest_cash_flow_statement["depreciation_and_amortization"]
delta_working_capital = latest_cash_flow_statement["change_in_working_capital"]
capex = latest_cash_flow_statement["capital_expenditure"]

fcff = (ebit + depreciation_and_amortization - income_tax_expense + delta_working_capital - capex)


st.write(f"""The FCFF is therefore:""")

# string interpolated formula for FCFF
fcff_formula = f"""
{fcff:,.0f} = {ebit:,.0f} + {depreciation_and_amortization:,.0f} - {income_tax_expense:,.0f} + {delta_working_capital:,.0f} - {capex:,.0f}
"""
st.latex(fcff_formula)

st.write(r"""
With a FCFF valuation model, one can typically choose between a single-stage, a two-stage, or a three-stage model.
         Single-stage FCFF model is useful for stable firms. Multi-stage models are useful for firms that are expected to grow at different rates in the future.
We use a two-stage FCFF valuation model here.""")

st.write(r"""
A general expression for the two-stage FCFF valuation model is:
         """)

st.latex(r"""
\text{Firm Value} = \sum_{t=1}^{n} \underbrace{\frac{FCFF_t}{(1+WACC)^t}}_{\text{Stage 1}} + \underbrace{\frac{FCFF_{n+1}}{(WACC - g)} \cdot {\frac{1}{(1+WACC)^n}}}_{\text{Stage 2}}
         """)

st.write(r"""
Where:
- $FCFF_t$ is the free cash flow to the firm in year $t$
- $WACC$ is the weighted average cost of capital
- $g$ is the long-term growth rate of FCFF after year $n$
- $n$ is the number of years in stage 1 (e.g., 5 years)
- $FCFF_{n+1}$ is the free cash flow to the firm in the last year (Year 6 in our case)
         """)

st.write("""## WACC (Weighted Average Cost of Capital)""")

st.write(r"""
The Weighted Average Cost of Capital (WACC) is the average rate of the reates of return required by each of the capital suppliers to the firm 
(equity and debt holders), where the weights are the proportions of each type of capital in the firm's capital structure.
         """)

st.latex(r"""
WACC = \frac{E}{D + E} \cdot r_e + \frac{D}{D + E} \cdot r_d \cdot (1 - \tau)
         """)

st.write(r"""
Where:
- $E$ is the market value of equity
- $D$ is the market value of debt
- $r_e$ is the cost of equity
- $r_d$ is the cost of debt
- $\tau$ is the effective tax rate
         """)

st.write("""### Required Return on Equity ($r_e$)""")

st.write(r"""We can use the Capital Asset Pricing Model (CAPM) to calculate the required return on equity ($r_e$):""")

st.latex(r"""
r_e = r_f + \beta \cdot (\text{ERP})
         """)

st.write(r"""
Where:
- $r_f$ is the risk-free rate (e.g., yield on 10-year Treasury bonds)
- $\beta$ is the stock's beta (a measure of its volatility relative to the market)
- $\text{ERP}$ is the equity risk premium (the expected return of the market minus the risk-free rate)
         """)



risk_free_rate = 0.0425  # Example: 4.25% risk-free rate
st.write(rf""" The risk-free rate for a US 10Y Treasury bond is {risk_free_rate * 100:.2f}%.
         """)
erp = 0.0433  # Example: 4.33% equity risk premium
st.write(rf""" We take the last estimated ERP from Damodaran's website, which is {erp * 100:.2f}%.""")


st.write("""We compute the betas using the CAPM model, using monthly returns and the last 5 years of data.""")

# load the betas from the CSV file
betas = pd.read_csv("./data/betas.csv")

selected_beta = betas[betas["symbol"] == selected_symbol]["estimate"].values[0]

st.write(f"""The beta for {selected_symbol} is:""")
st.latex(f"""\\beta = {selected_beta:.2f}
""")

# string interpolated equation for required return on equity
st.write(rf"""The required return on equity for {selected_symbol} is given by the formula:
         """)

st.latex(rf"""
r_e = {risk_free_rate * 100:.2f}\% + {selected_beta:.2f} \cdot ({erp * 100:.2f}\%)
""")

# string interpolated formula for required return on equity
required_return = risk_free_rate + selected_beta * erp
required_return_str = f"{required_return * 100:.2f}%"

st.write(f"""The required return on equity for {selected_symbol} is therefore {required_return_str}.
""")

st.write("""### Required Return on Debt ($r_d$)""")

st.write(r"""
         Firms have a cost of debt, which is the interest rate a company pays on its debt, such as bonds and loans. 
         From the lender's perspective, this rate is the required return on debt.

         We can obtain it directly from the firm's Interest Expense from the Income Statement and Total Debt from the Balance Sheet.

         After dividing the Interest Expense by Total Debt to calculate the required rate of return on debt, we need to multiply this rate 
         by (1 - tax rate) to account for the tax shield on interest payments. The reason for this is because there are tax deductions on interest paids. 
         As a result, the net cost of a company's debt is the amount of interesst the firm is paying minus the amount it has saved in taxes because of its 
         tax-deductible interest payments.
         """)

# load the cost of debt data from the CSV file
cost_of_debt = pd.read_csv("./data/cost_of_debt.csv")
selected_cost_of_debt = cost_of_debt[cost_of_debt["symbol"] == selected_symbol]

cost_of_debt = selected_cost_of_debt["cost_of_debt"].values[0]
tax_rate = selected_cost_of_debt["avg_effective_tax_rate"].values[0]
after_tax_cost_of_debt = selected_cost_of_debt["after_tax_cost_of_debt"].values[0]

st.latex(f"""
r_d = {cost_of_debt * 100:.2f}\% \cdot (1 - {tax_rate:.2f})
""")

st.latex(f"""
         = {after_tax_cost_of_debt * 100:.2f}\%
""")

st.write("""### Estimating the Capital Structure""")

st.write(r"""
         The final inputs for calculating WACC deal with the firm's capital structure, which is the company's mix of debt and equity financing.
         As previously noted, WACC is a blend of a company's equity and debt cost of capital, based on the company's equity and debt-capital ratio.


For the market value of equity, we can use the market capitalization of the firm, which is the stock price multiplied by the number of shares outstanding.

In most cases, we can use the book value of debt from a company's latest balance sheet as an approximation of the market value of debt. Unlike equity,
the market value of debt usually does not deviate too far from the book value. We take the Net Debt, which is the Total Debt minus Cash and Cash Equivalents from 
the Balance Sheet.
         """)

market_cap = pd.read_csv("./data/market_caps.csv")
# Load balance sheet statements
balance_sheet_statements = pd.read_csv("./data/balance_sheet_statements.csv")

selected_market_cap = market_cap[market_cap["symbol"] == selected_symbol]
selected_balance_sheet = balance_sheet_statements[balance_sheet_statements["symbol"] == selected_symbol]

# get the latest balance sheet data for the selected symbol
latest_balance_sheet = selected_balance_sheet.sort_values("calendar_year").iloc[-1]
net_debt = latest_balance_sheet["net_debt"]
market_cap = selected_market_cap["market_cap"].values[0]
we = selected_market_cap["market_cap"].values[0] / (selected_market_cap["market_cap"].values[0] + net_debt)
wd = net_debt / (selected_market_cap["market_cap"].values[0] + net_debt)


st.latex(f"""
w_e = \\frac{{{market_cap:,.0f}}}{{{market_cap:,.0f} + {net_debt:,.0f}}} = {we:.2f}
""")

st.latex(f"""
w_d = \\frac{{{net_debt:,.0f}}}{{{market_cap:,.0f} + {net_debt:,.0f}}} = {wd:.2f}
""")

st.write("""### Bringing it all together: WACC""")

st.write(r"""
         Now that we have all the components, we can calculate the WACC.
         """)

WACC = we * required_return + wd * after_tax_cost_of_debt
avg_effective_tax_rate = selected_cost_of_debt["avg_effective_tax_rate"].values[0]

# General WACC formula
st.latex(r"""
\text{WACC} = w_e \cdot r_e + w_d \cdot r_d (1 - \tau)
""")

# Numeric substitution
st.latex(
    fr"""
\text{{WACC}} =
{we * 100:.2f}\% \cdot {required_return * 100:.2f}\% +
{wd * 100:.2f}\% \cdot {cost_of_debt * 100:.2f}\% \times (1 - {avg_effective_tax_rate * 100:.2f}\%)
"""
)

# Final computed WACC value
st.latex(
    fr"""
\text{{WACC}} = {WACC * 100:.2f}\%
"""
)


st.write("""## Free Cash Flow to the Firm (FCFF)""")

st.write(""" ### Forecasting FCFF""")

st.write(r"""
         The next step is to forecast the FCFF for the next 5 years. 
         We can use the historical FCFF data to estimate the growth rate of FCFF, and then use that growth rate to forecast the FCFF for the next 5 years.

         We estimate the growth rate with a CAGR (Compound Annual Growth Rate) calculation.
         The long-term growth rate is the rate at which we expect the FCFF to grow after the first 5 years, assuming to be half of the CAGR.
         """)



# growth_forecast_data 
growth_forecast_data = (
    pd.read_csv("./data/growth_forecast.csv")
)


# Filter for the chosen symbol
symbol_data = growth_forecast_data[growth_forecast_data["symbol"] == selected_symbol].iloc[0]

# Parameters
WACC = 0.08
cagr = symbol_data["cagr"]
lt_growth = symbol_data["long_term_growth"]
n_years = 5  # forecast horizon
# Example last historical FCFF (you may pull from your real data)
FCFF0 = symbol_data["fcff"]

# Build LaTeX-safe strings for rates with escaped percent signs
wacc_str = f"{WACC*100:.2f}\\%"
cagr_str = f"{cagr*100:.2f}\\%"
ltg_str = f"{lt_growth*100:.2f}\\%"
fcff0_str = f"{FCFF0:,.0f}"  # no decimals, thousands separator

# Interpolated LaTeX formula with explicit growth
latex_formula = fr"""
\text{{Firm Value}} =
\sum_{{t=1}}^{{{n_years}}}
\underbrace{{\frac{{{fcff0_str} \times (1+{cagr_str})^t}}{{(1+{wacc_str})^t}}}}_{{\text{{Stage 1}}}}
+
\underbrace{{\frac{{{fcff0_str} \times (1+{cagr_str})^{{{n_years+1}}}}}{{({wacc_str} - {ltg_str})}}
\cdot \frac{{1}}{{(1+{wacc_str})^{n_years}}}}}_{{\text{{Stage 2}}}}
"""
# Display the LaTeX formula
st.latex(latex_formula)

# Optional: parameters summary
st.write(f"**CAGR (5Y):** {cagr:.2%}")
st.write(f"**Long-term growth rate:** {lt_growth:.2%}")
st.write(f"**WACC:** {WACC:.2%}")

# string interpolated formula for DCF where we put the value of cagr for growth rate 

# # Merge with long-term growth rate
# lt_growth = cagr_df.loc[cagr_df["symbol"] == selected_symbol, "long_term_growth"].values[0]

# WACC = 0.08  # 8% discount rate


# # --- Step 1: PV of forecast years ---
# fcff_forecast_selected["pv_fcff"] = fcff_forecast_selected.apply(
#     lambda row: row["forecast_fcff"] / ((1 + WACC) ** (row["year"] - latest_fcff.loc[latest_fcff["symbol"] == selected_symbol, "calendar_year"].values[0])),
#     axis=1
# )

# # --- Step 2: Terminal value ---
# fcff_year5 = fcff_forecast_selected.sort_values("year").iloc[-1]["forecast_fcff"]
# terminal_value = fcff_year5 * (1 + lt_growth) / (WACC - lt_growth)
# pv_terminal_value = terminal_value / ((1 + WACC) ** 5)

# # --- Step 3: Total DCF value ---
# dcf_value = fcff_forecast_selected["pv_fcff"].sum() + pv_terminal_value

# st.write(f"""
# The DCF value for {selected_symbol} is **${dcf_value:,.2f}**.
#          """)