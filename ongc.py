import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load stock and index data (adjust ticker as needed)
stock_ticker = "ONGC.NS"
index_ticker = "^NSEI"  # Nifty 50 as market index

# Fetch historical data
df_stock = yf.download(stock_ticker, start="2023-01-01", end="2024-03-31", auto_adjust=True)
df_index = yf.download(index_ticker, start="2023-01-01", end="2024-03-31", auto_adjust=True)

# Ensure 'Close' column is used since 'Adj Close' may not exist
df_stock["Returns"] = df_stock["Close"].pct_change()
df_index["Returns"] = df_index["Close"].pct_change()

# Drop NaN values
df = pd.concat([df_stock["Returns"], df_index["Returns"]], axis=1, keys=["Stock", "Market"]).dropna()

# Perform regression to get beta
slope, intercept, r_value, p_value, std_err = linregress(df["Market"], df["Stock"])
beta_equity = slope
print(f"Beta Equity: {beta_equity:.4f}")

# Constants for CAPM
risk_free_rate = 0.07  # Assume 7% as risk-free rate
market_return = 0.12  # Assume 12% as market return

# Calculate Cost of Equity
cost_of_equity = risk_free_rate + beta_equity * (market_return - risk_free_rate)
print(f"Cost of Equity: {cost_of_equity:.4f}")

# Cost of Debt Estimation (Using Interest Coverage Ratio)
interest_expense = 102895440  # From financials
EBIT = 680222690  # From financials

interest_coverage_ratio = EBIT / interest_expense
if interest_coverage_ratio > 8:
    cost_of_debt = 0.05  # 5% for high rating
elif interest_coverage_ratio > 5:
    cost_of_debt = 0.06  # 6% for medium rating
else:
    cost_of_debt = 0.08  # 8% for low rating
print(f"Cost of Debt: {cost_of_debt:.4f}")

# Market Value of Equity
if df_stock["Close"].dropna().empty:
    raise ValueError("Stock price data is missing. Check your data source.")

stock_price = df_stock["Close"].dropna().iloc[-1]
shares_outstanding = 125820190  # From financials
market_value_equity = float(stock_price) * shares_outstanding  # Ensure it's a scalar value
if np.isnan(market_value_equity) or market_value_equity <= 0:
    raise ValueError("Market Value of Equity calculation failed. Check stock price and shares outstanding data.")
print(f"Market Value of Equity: {market_value_equity:,.2f}")

# Market Value of Debt Estimation
book_value_debt = 500000000  # Assumed from financials
market_value_debt = book_value_debt * 1.1  # Assume 10% premium
if np.isnan(market_value_debt) or market_value_debt <= 0:
    raise ValueError("Market Value of Debt calculation failed. Check book value assumptions.")
print(f"Market Value of Debt: {market_value_debt:,.2f}")

# WACC Calculation
corporate_tax_rate = 0.3  # 30% assumed tax rate

total_value = market_value_equity + market_value_debt
wacc = (market_value_equity / total_value) * cost_of_equity + (market_value_debt / total_value) * cost_of_debt * (1 - corporate_tax_rate)
print(f"Weighted Average Cost of Capital (WACC): {wacc:.4f}")

# Plot Stock vs Market Returns
plt.figure(figsize=(10, 6))
plt.scatter(df["Market"], df["Stock"], alpha=0.5, label="Stock vs Market Returns")
plt.plot(df["Market"], intercept + beta_equity * df["Market"], color='red', label="Regression Line")
plt.xlabel("Market Returns")
plt.ylabel("Stock Returns")
plt.title("Stock vs Market Returns and Beta Regression")
plt.legend()
plt.show()

# Plot Stock Price Movement
df_stock["Close"].plot(figsize=(10, 6), title="Stock Price Over Time", grid=True)
plt.ylabel("Stock Price")
plt.show()
