import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import linregress

st.title("WACC Calculator")

# User input for stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g., ONGC.NS):", "ONGC.NS")

# Fetch historical data
def fetch_data(ticker):
    try:
        df = yf.download(ticker, start="2023-01-01", end="2024-03-31", auto_adjust=True)
        if df.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

df_stock = fetch_data(stock_ticker)
index_ticker = "^NSEI"
df_index = fetch_data(index_ticker)

# Ensure 'Close' column is used since 'Adj Close' may not exist
df_stock["Returns"] = df_stock["Close"].pct_change()
df_index["Returns"] = df_index["Close"].pct_change()

# Drop NaN values
df = pd.concat([df_stock["Returns"], df_index["Returns"]], axis=1, keys=["Stock", "Market"]).dropna()

# Perform regression to get beta
slope, intercept, r_value, p_value, std_err = linregress(df["Market"], df["Stock"])
beta_equity = slope
st.write(f"**Beta Equity:** {beta_equity:.4f}")

# Constants for CAPM
risk_free_rate = 0.07  # Assume 7% as risk-free rate
market_return = 0.12  # Assume 12% as market return

# Calculate Cost of Equity
cost_of_equity = risk_free_rate + beta_equity * (market_return - risk_free_rate)
st.write(f"**Cost of Equity:** {cost_of_equity:.4f}")

# Market Value of Equity
stock_price = float(df_stock["Close"].dropna().iloc[-1])
shares_outstanding = 125820190.0  # Fixed value
market_value_equity = stock_price * shares_outstanding
st.write(f"**Market Value of Equity:** {market_value_equity:,.2f}")

# Market Value of Debt Estimation
book_value_debt = 500000000.0  # Fixed value
market_value_debt = book_value_debt * 1.1  # Assume 10% premium
st.write(f"**Market Value of Debt:** {market_value_debt:,.2f}")

# WACC Calculation
corporate_tax_rate = 0.3  # Fixed value
cost_of_debt = 0.06  # Fixed value
total_value = market_value_equity + market_value_debt
wacc = (market_value_equity / total_value) * cost_of_equity + (market_value_debt / total_value) * cost_of_debt * (1 - corporate_tax_rate)
st.write(f"**Weighted Average Cost of Capital (WACC):** {wacc:.4f}")

# Plot Stock vs Market Returns
st.subheader("Stock vs Market Returns and Beta Regression")
fig, ax = plt.subplots()
ax.scatter(df["Market"], df["Stock"], alpha=0.5, label="Stock vs Market Returns")
ax.plot(df["Market"], intercept + beta_equity * df["Market"], color='red', label="Regression Line")
ax.set_xlabel("Market Returns")
ax.set_ylabel("Stock Returns")
ax.set_title("Stock vs Market Returns and Beta Regression")
ax.legend()
st.pyplot(fig)

# Plot Stock Price Movement
st.subheader("Stock Price Over Time")
fig, ax = plt.subplots()
df_stock["Close"].plot(ax=ax, title="Stock Price Over Time", grid=True)
ax.set_ylabel("Stock Price")
st.pyplot(fig)
