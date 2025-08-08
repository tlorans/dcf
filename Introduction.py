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
WACC = \frac{E}{D + E} \cdot r_e + \frac{D}{D + E} \cdot r_d
         """)

st.write(r"""
Where:
- $E$ is the market value of equity
- $D$ is the market value of debt
- $r_e$ is the cost of equity
- $r_d$ is the cost of debt

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

beta_series = betas.loc[betas["symbol"] == selected_symbol, "estimate"]

if beta_series.empty:
    st.warning(f"No beta found for {selected_symbol}. Using β = 1.00 as a fallback.")
    selected_beta = 1.0
else:
    selected_beta = float(beta_series.iloc[0])


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
st.write("### Inferring the Cost and Market Value of Debt from Equity Prices")

st.write(r"""
We start from **observed market capitalization** $E$ (equity value) and the **face value of debt** $D$ (from the balance sheet or bond data).  
We do **not** assume enterprise value in advance. Instead, we invert a Merton-style structural model to recover the **implied asset value** $V$ that is consistent with the observed equity price.

---

**Step 1 — Merton model for equity**

In the Merton framework, equity is a call option on the firm's assets with strike price equal to the face value of debt:

$E = V \cdot N(d_1) - D \cdot e^{-r_f T} \, N(d_2)$

where:

$d_1 = \frac{\ln(V/D) + \left(r_f + \tfrac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}},$  
$d_2 = d_1 - \sigma\sqrt{T}$

---

**Step 2 — Solving for asset value**

We fix the observed $E$ and solve for $V$ such that the Merton pricing equation holds.  
This $V$ represents the market value of the firm's assets, consistent with the market price of equity.

---

**Step 3 — Probability of default and cost of debt**

From $d_2$, the **risk-neutral probability of default** over horizon $T$ is:

$\pi = 1 - N(d_2)$

Given a loss-given-default $L$, the (pre-tax) **cost of debt** follows:

$r_d = (1-\pi) r_f + \pi L$

so that the **default spread** is:

$\text{Default spread} = r_d - r_f$

---

**Step 4 — Market value of debt**

The market value of debt is the residual:

$D_{\text{market}} = V - E$

Because $\pi > 0$, we generally have:

$D_{\text{market}} < D_{\text{face}}$

reflecting the discount investors demand for bearing default risk.

---

We use Damodaran's total market standard deviation of firm value to proxy $\sigma$.
""")


sigma = 0.2922  # Example: 29.22% annualized firm value volatility

market_cap = pd.read_csv("./data/market_caps.csv")
# Load balance sheet statements
balance_sheet_statements = pd.read_csv("./data/balance_sheet_statements.csv")

selected_market_cap = market_cap[market_cap["symbol"] == selected_symbol]
selected_balance_sheet = balance_sheet_statements[balance_sheet_statements["symbol"] == selected_symbol]

# load the cost of debt data from the CSV file
cost_of_debt = pd.read_csv("./data/cost_of_debt.csv")
selected_cost_of_debt = cost_of_debt[cost_of_debt["symbol"] == selected_symbol]

# get the latest balance sheet data for the selected symbol
latest_balance_sheet = selected_balance_sheet.sort_values("calendar_year").iloc[-1]
net_debt = latest_balance_sheet["net_debt"]
market_cap = selected_market_cap["market_cap"].values[0]
avg_effective_tax_rate = selected_cost_of_debt["avg_effective_tax_rate"].values[0]



import numpy as np
from math import log, sqrt, exp
try:
    from scipy.stats import norm
    cnd = norm.cdf
except:
    # Simple CND fallback if SciPy isn't available
    import math
    def cnd(x):
        return 0.5 * (1 + math.erf(x / sqrt(2)))

# Inputs from your app (already defined earlier)
E = float(market_cap)                            # market equity value
D = float(net_debt if net_debt > 0 else 0.0)     # debt proxy (use better face value if you have it)
r = float(risk_free_rate)
T = 10.0
LGD = 1.

st.latex(fr"""
d_1 = \frac{{\ln\left(\frac{{{E+D:,.0f}}}{{{D:,.0f}}}\right) + \left({r*100:.2f}\% + \tfrac{{1}}{{2}}\cdot{{{sigma:.4f}}}^2\right)\cdot{{{T}}}}}{{{{{sigma:.4f}}}\cdot\sqrt{{{T}}}}}
""")


from scipy.optimize import fsolve

# --- Step 1: Inputs ---
rf = r            # Risk-free rate
T = 10.0          # Horizon (years)
LGD = 1.0         # Loss given default (100%)
E_obs = E         # Observed equity value
D_face = D        # Face value of debt (from balance sheet)

# --- Step 2: Merton model functions ---
def merton_equity_value(V):
    """Merton equity valuation given asset value V."""
    d1 = (log(V / D_face) + (rf + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return V * cnd(d1) - D_face * exp(-rf * T) * cnd(d2)

# --- Step 3: Solve for asset value V so E_model = E_obs ---
V_initial_guess = E_obs + D_face
V_solution = fsolve(lambda V: merton_equity_value(V) - E_obs, V_initial_guess)[0]

# --- Step 4: Compute d1, d2, PD ---
d1 = (log(V_solution / D_face) + (rf + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
PD = 1 - cnd(d2)  # Risk-neutral probability of default

# --- Step 5: Cost of debt ---
r_d = (1 - PD) * rf + PD * LGD
default_spread = r_d - rf

# --- Step 6: Market value of debt ---
D_market = V_solution - E_obs


st.latex(fr"""
d_2 = {d2:.4f},\quad
\pi = 1 - N(d_2) = {PD:.2%}
""")

st.latex(fr"""
r_d = (1 - \pi) \cdot {rf * 100:.2f}\% + {PD * 100:.2f} \cdot {LGD * 100:.2f}\% = {r_d * 100:.2f}\%
""")

st.latex(fr"""
D_{{\text{{market}}}} = {V_solution:,.0f} - {E_obs:,.0f} = {D_market:,.0f}
""")

st.write("""### Bringing it all together: WACC""")

st.write(r"""
         The final inputs for calculating WACC deal with the firm's capital structure, which is the company's mix of debt and equity financing.
         As previously noted, WACC is a blend of a company's equity and debt cost of capital, based on the company's equity and debt-capital ratio.


For the market value of equity, we can use the market capitalization of the firm, which is the stock price multiplied by the number of shares outstanding.

For the market value of debt, we can use the market value of debt that we just calculated.
         """)

we = selected_market_cap["market_cap"].values[0] / (selected_market_cap["market_cap"].values[0] + D_market)
wd = D_market / (selected_market_cap["market_cap"].values[0] + D_market)


st.latex(f"""
w_e = \\frac{{{market_cap:,.0f}}}{{{market_cap:,.0f} + {D_market:,.0f}}} = {we:.2f}
""")

st.latex(f"""
w_d = \\frac{{{D_market:,.0f}}}{{{market_cap:,.0f} + {D_market:,.0f}}} = {wd:.2f}
""")


st.write(r"""
         Now that we have all the components, we can calculate the WACC.
         """)

WACC = we * required_return + wd * r_d

# General WACC formula
st.latex(r"""
\text{WACC} = w_e \cdot r_e + w_d \cdot r_d
""")

# Numeric substitution
st.latex(
    fr"""
\text{{WACC}} =
{we * 100:.2f}\% \cdot {required_return * 100:.2f}\% +
{wd * 100:.2f}\% \cdot {r_d * 100:.2f}\%
"""
)

# Final computed WACC value
st.latex(
    fr"""
\text{{WACC}} = {WACC * 100:.2f}\%
"""
)

st.write("## Reverse-engineering the Implied Growth Rate")

st.write(r"""
We can work backwards from the **current market price** (or Enterprise Value) to determine the growth rate 
that the market is implicitly assuming in the first stage of the DCF.

We assume:
- A constant annual growth rate $g$ for the first $n$ years (Stage 1),
- A long-term growth rate $g_{\text{LT}} = g / 2$ thereafter (Stage 2),
- A Weighted Average Cost of Capital $WACC$.

The **implied growth rate** is the value of $g$ that satisfies:
""")

st.latex(r"""
     \underbrace{
\sum_{t=1}^{n} \frac{FCFF_0 \cdot (1+g)^t}{(1+WACC)^t}
}_{\text{Stage 1}}
\;+\;
\underbrace{
\frac{FCFF_0 \cdot (1+g)^{n+1}}{WACC - g/2} \cdot \frac{1}{(1+WACC)^n}
}_{\text{Stage 2}}
= \text{EV}
         """)


# growth_forecast_data 
growth_forecast_data = (
    pd.read_csv("./data/growth_forecast.csv")
)


# Filter for the chosen symbol
symbol_data = growth_forecast_data[growth_forecast_data["symbol"] == selected_symbol].iloc[0]


# cagr = symbol_data["cagr"]
# lt_growth = symbol_data["long_term_growth"]
n_years = 5  # forecast horizon
# Example last historical FCFF (you may pull from your real data)
FCFF0 = symbol_data["fcff"]

# --- Implied growth with LT floor: g can be < 0, but g_LT = max(g/2, 0) ---

def firm_value_from_g(FCFF0, WACC, n, g):
    # Disallow impossible compounding
    if (1 + g) <= 0:
        return np.inf

    # Enforce non-negative long-term growth
    g_lt = g / 2.0
    if g_lt < 0.0:
        g_lt = 0.0

    # Terminal condition
    if (WACC - g_lt) <= 0:
        return np.inf

    # Stage 1 (years 1..n)
    r = (1 + g) / (1 + WACC)
    if abs(r - 1) < 1e-12:
        stage1 = FCFF0 * n / (1 + WACC)
    else:
        stage1 = FCFF0 * r * (1 - r**n) / (1 - r)

    # Stage 2 (terminal)
    tv = FCFF0 * (1 + g)**(n + 1) / (WACC - g_lt)
    stage2 = tv / (1 + WACC)**n

    return stage1 + stage2

def implied_cagr_from_price(FCFF0, WACC, n, EV, g_min=-0.9, g_max=None, tol=1e-8, max_iter=200):
    """
    Finds g such that FirmValue(g) = EV with:
      - Stage-1 growth g free to be negative
      - Long-term growth g_LT = max(g/2, 0) >= 0
    """
    # Upper bound must respect g_LT < WACC -> g < 2*WACC when g >= 0
    if g_max is None:
        g_max = min(2*WACC - 1e-6, 0.5)  # also cap at +50% for sanity

    # Evaluate f(g) = PV(g) - EV
    def f(g):
        return firm_value_from_g(FCFF0, WACC, n, g) - EV

    # Robust bracketing: scan a grid for a sign change
    grid = np.linspace(g_min, g_max, 61)  # 60 intervals
    vals = []
    for x in grid:
        fx = f(x)
        vals.append(fx if np.isfinite(fx) else np.nan)

    # find first adjacent finite sign change
    bracket = None
    for i in range(len(grid)-1):
        a, b = grid[i], grid[i+1]
        fa, fb = vals[i], vals[i+1]
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            bracket = (a, b, fa, fb)
            break

    if bracket is None:
        return np.nan  # no solution under constraints

    g_low, g_high, f_low, f_high = bracket

    # Bisection
    for _ in range(max_iter):
        g_mid = 0.5 * (g_low + g_high)
        f_mid = f(g_mid)
        if not np.isfinite(f_mid):
            # shrink towards finite side
            g_high = g_mid
            continue
        if abs(f_mid) < tol:
            return g_mid
        if f_low * f_mid <= 0:
            g_high, f_high = g_mid, f_mid
        else:
            g_low, f_low = g_mid, f_mid

    return g_mid  # best effort


# --- Use it with your current variables ---
EV = float(market_cap + D_market)  # enterprise value (equity + net debt)

implied_g = implied_cagr_from_price(
    FCFF0=float(FCFF0),
    WACC=float(WACC),
    n=int(n_years),
    EV=EV
)

if FCFF0 <= 0:
    st.warning("FCFF0 is non-positive; implied growth cannot be solved under the Gordon setup.")


if np.isnan(implied_g):
    
    st.warning("Could not find an implied growth that matches the current price with these inputs.")
else:
    implied_lt = max(implied_g/2.0, 0.0)
    # st.write("### Implied Growth to Match Current Price")
    # st.write(f"- **Implied CAGR (Years 1–{n_years})**: {implied_g:.2%}")
    # st.write(f"- **Implied Long-term growth (from Year {n_years+1})**: {implied_lt:.2%}")

    # Show the numeric substitution DCF with implied g
    wacc_str = f"{WACC*100:.2f}\\%"
    g_str = f"{implied_g*100:.2f}\\%"
    glt_str = f"{implied_lt*100:.2f}\\%"
    fcff0_str = f"{FCFF0:,.0f}"
    n_str = str(n_years)

    st.latex(fr"""
\text{{Firm Value}} =
\sum_{{t=1}}^{{{n_str}}}
\frac{{{fcff0_str}\cdot (1+{g_str})^t}}{{(1+{wacc_str})^t}}
+\;
\frac{{{fcff0_str}\cdot (1+{g_str})^{{{int(n_years)+1}}}}}{{{wacc_str} - {glt_str}}}
\cdot \frac{{1}}{{(1+{wacc_str})^{{{n_str}}}}}
\;=\; {EV:,.0f}
""")


    # Implied CAGR and long-term growth in equation form
    st.latex(fr"""
    g  = {g_str},
    \quad
    g_{{\text{{LT}}}} = \frac{{g}}{2} = {glt_str}
    """)

st.write(r"""
## Stress Testing the Model

We can also stress test the model by varying the inputs and observing how the valuation changes.
""")


import numpy as np
from math import log, sqrt, exp
try:
    from scipy.stats import norm
    cnd = norm.cdf
except Exception:
    import math
    def cnd(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def dcf_value_fcff(FCFF0, g, n, WACC):
    """Two-stage FCFF DCF with LT growth floor: g_LT = max(g/2, 0)."""
    if (1 + g) <= 0:
        return np.inf
    g_lt = max(g/2.0, 0.0)
    if (WACC - g_lt) <= 0:
        return np.inf

    r = (1 + g) / (1 + WACC)
    # Stage 1
    if abs(r - 1) < 1e-12:
        stage1 = FCFF0 * n / (1 + WACC)
    else:
        stage1 = FCFF0 * r * (1 - r**n) / (1 - r)
    # Stage 2
    tv = FCFF0 * (1 + g)**(n + 1) / (WACC - g_lt)
    stage2 = tv / (1 + WACC)**n
    return stage1 + stage2

def merton_from_V(V, D_face, rf, sigma, T):
    """Given asset value V, compute E (equity), d1,d2, PD under Merton."""
    if V <= 0 or D_face <= 0 or sigma <= 0 or T <= 0:
        return np.nan, np.nan, np.nan, np.nan
    d1 = (log(V / D_face) + (rf + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    E = V * cnd(d1) - D_face * exp(-rf * T) * cnd(d2)
    PD = 1.0 - cnd(d2)
    return E, d1, d2, PD

def kd_from_PD(PD, rf, LGD):
    """Pre-tax cost of debt from PD and LGD."""
    return (1 - PD) * rf + PD * LGD

def wacc_from_components(E, Dmkt, ke, kd):
    """Market-value weighted WACC."""
    total = E + Dmkt
    if total <= 0:
        return np.nan, np.nan, np.nan
    we = E / total
    wd = Dmkt / total
    return we * ke + wd * kd, we, wd

def stress_solve_single_g(
    g, FCFF0, n, rf, ke, D_face, sigma, T,
    LGD=1.0, wacc_init=None, tol=1e-7, max_iter=100
):
    """
    Fixed-point iteration:
      Start with WACC, DCF -> V
      Merton at that V -> (E, PD, D_market)
      PD -> k_d; (E,D_market,ke,kd) -> new WACC
      Repeat until convergence.
    """
    if wacc_init is None:
        # Conservative starting guess: ke weighted with book D
        # (You can also reuse the WACC you computed earlier.)
        wacc_init = ke * 0.7 + (rf + 0.02) * 0.3  # simple fallback
    WACC = float(wacc_init)

    V_prev = None
    for _ in range(max_iter):
        # 1) DCF with current WACC
        V = dcf_value_fcff(FCFF0, g, n, WACC)
        if not np.isfinite(V):
            return {"ok": False, "msg": "DCF invalid (rates/growth combo)."}
        # 2) Merton given V
        E, d1, d2, PD = merton_from_V(V, D_face, rf, sigma, T)
        if not np.isfinite(E) or E <= 0:
            return {"ok": False, "msg": "Merton invalid (E non-positive)."}
        Dmkt = V - E
        if Dmkt <= 0:
            return {"ok": False, "msg": "Implied debt market value non-positive."}
        # 3) k_d from PD
        kd = kd_from_PD(PD, rf, LGD)
        # 4) New WACC
        WACC_new, we, wd = wacc_from_components(E, Dmkt, ke, kd)
        if not np.isfinite(WACC_new):
            return {"ok": False, "msg": "WACC invalid."}

        # Convergence checks on both WACC and V
        if V_prev is not None and abs(V - V_prev) < tol and abs(WACC_new - WACC) < tol:
            return {
                "ok": True,
                "g": g,
                "V": V,
                "E": E,
                "Dmkt": Dmkt,
                "PD": PD,
                "kd": kd,
                "WACC": WACC_new,
                "we": we,
                "wd": wd,
                "d1": d1,
                "d2": d2
            }
        V_prev = V
        WACC = WACC_new

    return {"ok": False, "msg": "Did not converge within max_iter."}


# ---------- Quick Stress: scale implied_g only (all else fixed) ----------
st.write("## Quick Stress: Scale the Implied Growth (all other inputs fixed)")

if np.isnan(implied_g):
    st.warning("Implied growth (g) wasn't found above, so I can't run the scaling stress. Check inputs/EV.")
else:
    # Choose a simple scale for g: e.g., 80% of implied_g
    scale_pct = st.slider("Scale factor for implied g (%)", 0, 150, 80, 5)
    scale_mult = scale_pct / 100.0
    new_g = float(implied_g) * scale_mult
    new_g_lt = max(new_g / 2.0, 0.0)  # keep your non-negative LT rule

    # Keep ALL other parameters exactly as already set
    rf = float(risk_free_rate)
    ke = float(required_return)
    FCFF0 = float(FCFF0)  # or float(fcff) if that's your last FCFF
    D_face = float(net_debt if net_debt > 0 else 1e-6)
    sigma_used = float(sigma)  # same sigma as above
    T_used = float(T)          # same T as above
    LGD_used = float(LGD)      # same LGD as above
    wacc_init_guess = float(WACC)  # start from your current WACC

    res = stress_solve_single_g(
        g=new_g,
        FCFF0=FCFF0,
        n=int(n_years),
        rf=rf,
        ke=ke,
        D_face=D_face,
        sigma=sigma_used,
        T=T_used,
        LGD=LGD_used,
        wacc_init=wacc_init_guess,
        tol=1e-7,
        max_iter=200
    )

 


    if not res["ok"]:
        st.warning(f"Scaled g = {new_g:.2%}: {res['msg']}")
    else:
        st.write("### Scaled scenario results")
        st.latex(fr"g = {new_g*100:.2f}\%, \quad g_{{\text{{LT}}}} = \max(g/2,0) = {new_g_lt*100:.2f}\%")
        st.write({
            "V (EV)": f"{res['V']:,.0f}",
            "E (Equity)": f"{res['E']:,.0f}",
            "D_market": f"{res['Dmkt']:,.0f}",
            "PD": f"{res['PD']:.2%}",
            "k_d": f"{res['kd']*100:.2f}%",
            "WACC": f"{res['WACC']*100:.2f}%",
            "w_e": f"{res['we']*100:.1f}%",
            "w_d": f"{res['wd']*100:.1f}%"
        })
    # result of new equity valuation vs market cap
        percentage_change_EV = (res["V"] - EV) / EV * 100.0
        percentage_change_equity = (res["E"] - market_cap) / market_cap * 100.0

        st.write(f"Percentage change in EV from scaling g: {percentage_change_EV:.2f}%")
        st.write(f"Percentage change in Equity from scaling g: {percentage_change_equity:.2f}%")
