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
The formal definition of FCFF is the cash available to all of the firm's investors, including stockholders and bonholders, after the firm buys and sells products,
         provides services, pays its cash operating expenses and makes short-term and long-term investments.
         """)


st.write(r"""
With a FCFF valuation model, one can typically choose between a singel-stage, a two-stage, or a three-stage model.
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


st.write("""We compute the betas for the Dow Jones Industrial Average constituents using the CAPM model, using monthly returns and the last 5 years of data.""")

st.image("./images/betas.png", caption="Betas for Dow Jones Industrial Average constituents")

# load the betas from the CSV file
betas = pd.read_csv("./data/betas.csv")

# Calculate the required return on equity for each stock
betas["required_return"] = risk_free_rate + betas["estimate"] * erp
# Sort the betas by required return
betas = betas.sort_values(by="required_return", ascending=True)

# Make symbol a categorical variable with the correct order
betas["symbol"] = pd.Categorical(
    betas["symbol"],
    categories=betas["symbol"],  # keeps the sorted order
    ordered=True
)

# Plot
required_return_plot = (
    ggplot(betas, aes(x="symbol", y="required_return")) +
    geom_col() +
    scale_y_continuous(labels=percent_format()) +
    coord_flip() +
    labs(
        x="Required Return on Equity", y="", 
        title="Required Return on Equity"
    )
)

required_return_plot.save("./images/required_return.png", dpi=300)

st.image("./images/required_return.png", caption="Required Return on Equity for Dow Jones Industrial Average constituents")

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

cost_of_debt = cost_of_debt.sort_values(by="after_tax_cost_of_debt", ascending=True)


# plot the column after_tax_cost_of_debt as the required return on debt
cost_of_debt["symbol"] = pd.Categorical(
    cost_of_debt["symbol"],
    categories=cost_of_debt["symbol"],  # keeps the sorted order
    ordered=True
)

cost_of_debt_plot = (
    ggplot(cost_of_debt, aes(x="symbol", y="after_tax_cost_of_debt")) +
    geom_col() +
    scale_y_continuous(labels=percent_format()) +
    coord_flip() +
    labs(
        x="Required Return on Debt", y="",
        title="Required Return on Debt"
    )
)
cost_of_debt_plot.save("./images/cost_of_debt.png", dpi=300)

st.image("./images/cost_of_debt.png", caption="Required Return on Debt")

st.write("""### Estimating the Capital Structure""")

st.write(r"""
         The final inputs for calculating WACC deal with the firm's capital structure, which is the company's mix of debt and equity financing.
         As previously noted, WACC is a blend of a company's equity and debt cost of capital, based on the company's equity and debt-capital ratio.


For the market value of equity, we can use the market capitalization of the firm, which is the stock price multiplied by the number of shares outstanding.

In most cases, we can use the book value of debt from a company's latest balance sheet as an approximation of the market value of debt. Unlike equity,
the market value of debt usually does not deviate too far from the book value. We take the Net Debt, which is the Total Debt minus Cash and Cash Equivalents from 
the Balance Sheet.
         """)

# load the capital structure data from the CSV file
capital_structure = pd.read_csv("./data/capital_structure.csv")

# If not already there, compute equity weight as the residual
if "w_equity" not in capital_structure.columns:
    capital_structure["w_equity"] = 1 - capital_structure["w_debt"]

# Reshape to long format for stacked bars
capital_structure_long = capital_structure.melt(
    id_vars=["symbol"],
    value_vars=["w_debt", "w_equity"],
    var_name="component",
    value_name="weight"
)

# Sort by debt weight (so order is consistent)
order_symbols = (
    capital_structure.sort_values(by="w_debt", ascending=True)["symbol"]
)
capital_structure_long["symbol"] = pd.Categorical(
    capital_structure_long["symbol"],
    categories=order_symbols,
    ordered=True
)

# Plot stacked bar chart
capital_structure_plot = (
    ggplot(capital_structure_long, aes(x="symbol", y="weight", fill="component")) +
    geom_col() +
    scale_y_continuous(labels=percent_format()) +
    coord_flip() +
    scale_fill_manual(
        values={"w_debt": "#d95f02", "w_equity": "#1b9e77"},
        labels={"w_debt": "Debt", "w_equity": "Equity"}
    ) +
    labs(
        x="", y="",
        fill="Component",
        title="Capital Structure: Debt vs Equity"
    )
)

capital_structure_plot.save("./images/capital_structure_stacked.png", dpi=300)

st.image(
    "./images/capital_structure_stacked.png",
    caption="Capital Structure (Debt vs Equity) of Dow Jones Industrial Average constituents"
)