import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Brent Crude Oil Forecasting Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Title
st.title("🛢️ Brent Crude Oil Price Forecasting Dashboard")
st.markdown("*Forecasting model built with Facebook Prophet | Data from Yahoo Finance*")

# Load data
@st.cache_data
def load_data():
    oil = pd.read_csv("C:/Users/DEEBYTE COMPUTERS/Documents/oil-forecasting-pipeline/data/oil_prices_cleaned.csv", index_col=0, parse_dates=True)
    oil.columns = ["Price", "Year", "Month", "Day", "MA_7", "MA_30", "MA_90", "Daily_Change", "Pct_Change", "Volatility_30"]
    return oil

@st.cache_data
def load_forecast():
    forecast = pd.read_csv("C:/Users/DEEBYTE COMPUTERS/Documents/oil-forecasting-pipeline/data/forecast.csv", parse_dates=["ds"])
    return forecast

load_data.clear()
load_forecast.clear()
oil = load_data()
forecast = load_forecast()

# Key metrics row
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

current_price = oil["Price"].iloc[-1]
predicted_price = forecast["yhat"].iloc[-30]
max_price = oil["Price"].max()
min_price = oil["Price"].min()

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Next Day Forecast", f"${predicted_price:.2f}", f"{predicted_price - current_price:.2f}")
col3.metric("All Time High", f"${max_price:.2f}")
col4.metric("All Time Low", f"${min_price:.2f}")

st.divider()

# Historical price chart
st.subheader("📈 Historical Oil Prices with Moving Averages")
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(oil.index, oil["Price"], color="darkorange", linewidth=0.8, label="Daily Price", alpha=0.8)
ax1.plot(oil.index, oil["MA_30"], color="blue", linewidth=1.5, label="30 Day MA")
ax1.plot(oil.index, oil["MA_90"], color="red", linewidth=2, label="90 Day MA")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.divider()

# Forecast chart with slider
# Forecast chart with slider
st.subheader("🔮 Price Forecast")
forecast_days = st.slider("Select forecast horizon (days)", 30, 700, 150)
#forecast_days = st.slider("Select forecast horizon (days)", 30, 298, 150)

# Get the last actual date
last_actual_date = oil.index[-1]

# Everything up to today
historical_forecast = forecast[forecast["ds"] <= last_actual_date]

# Future predictions only
future_forecast = forecast[forecast["ds"] > last_actual_date].head(forecast_days)

# Combine
forecast_filtered = pd.concat([historical_forecast, future_forecast])

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(oil.index, oil["Price"], color="darkorange", linewidth=0.8, label="Actual Price", alpha=0.8)
ax2.plot(forecast_filtered["ds"], forecast_filtered["yhat"], color="blue", linewidth=1.5, label="Forecast")
ax2.fill_between(forecast_filtered["ds"], forecast_filtered["yhat_lower"], forecast_filtered["yhat_upper"], alpha=0.2, color="blue", label="Uncertainty Range")
ax2.axvline(x=last_actual_date, color="red", linestyle="--", label="Today")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Volatility chart
st.subheader("⚡ Market Volatility (30 Day Rolling)")
fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(oil.index, oil["Volatility_30"], color="purple", linewidth=0.8)
ax3.fill_between(oil.index, oil["Volatility_30"], alpha=0.3, color="purple")
ax3.set_xlabel("Date")
ax3.set_ylabel("Volatility")
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.divider()

# Footer
st.markdown("*Built by Fatima Javed | Oil Price Forecasting Pipeline | 2026*")
