import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
from retrying import retry

st.title("Live SPY and VIX Analysis")

# Retry decorator for handling rate limit errors
def retry_if_rate_limit(exception):
    return isinstance(exception, yf.exceptions.YFRateLimitError)

@retry(retry_on_exception=retry_if_rate_limit, stop_max_attempt_number=5, wait_fixed=2000)
def fetch_ticker_data(ticker, period, interval=None):
    try:
        if interval:
            data = yf.download(ticker, period=period, interval=interval)
        else:
            data = yf.Ticker(ticker).history(period=period)
        time.sleep(1)  # Add delay to avoid rapid requests
        return data
    except yf.exceptions.YFRateLimitError as e:
        st.warning("Rate limit hit. Retrying...")
        raise e
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Fetch historical data for SPY and VIX (cached)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data():
    spy = fetch_ticker_data("SPY", period="10y", interval="1d")
    vix = fetch_ticker_data("^VIX", period="10y", interval="1d")
    if spy is not None and vix is not None:
        spy['Returns'] = spy['Close'].pct_change()
        vix['Returns'] = vix['Close'].pct_change()
    return spy, vix

# Fetch latest SPY and VIX data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_latest_data():
    latest_spy = fetch_ticker_data("SPY", period="1d")
    latest_vix = fetch_ticker_data("^VIX", period="1d")
    return latest_spy, latest_vix

# Load data
spy_data, vix_data = fetch_historical_data()
latest_spy, latest_vix = fetch_latest_data()

# Check if data was fetched successfully
if spy_data is None or vix_data is None or latest_spy is None or latest_vix is None:
    st.error("Failed to fetch data. Please try again later.")
else:
    # Extract latest prices and dates
    current_spy_price = latest_spy['Close'].iloc[-1]
    current_spy_date = latest_spy.index[-1].strftime("%Y-%m-%d")
    current_vix_price = latest_vix['Close'].iloc[-1]
    current_vix_date = latest_vix.index[-1].strftime("%Y-%m-%d")

    # VIX spike analysis
    threshold_2sd = vix_data['Close'].mean() + 2 * vix_data['Close'].std()
    threshold_3sd = vix_data['Close'].mean() + 3 * vix_data['Close'].std()

    vix_data['Spike Level'] = np.where(
        vix_data['Close'] >= threshold_3sd, '3SD',
        np.where(vix_data['Close'] >= threshold_2sd, '2SD', 'No Spike')
    )

    vix_data['Month'] = vix_data.index.month
    vix_data['Day'] = vix_data.index.day

    # Calculate VIX percentile rank
    sorted_vix_prices = np.sort(vix_data['Close'])
    vix_percentile = (sorted_vix_prices < current_vix_price).sum() / len(sorted_vix_prices) * 100

    # Display data table
    st.subheader("Current SPY & VIX Prices, Threshold Levels, and VIX Percentile Rank")
    price_levels = pd.DataFrame({
        "Metric": ["Current SPY Price", "Current VIX Price", "2 Standard Deviations (VIX)", "3 Standard Deviations (VIX)", "VIX Percentile Rank"],
        "Value": [
            float(current_spy_price),
            float(current_vix_price),
            float(threshold_2sd),
            float(threshold_3sd),
            f"{vix_percentile:.2f}th Percentile"
        ],
        "Date": [current_spy_date, current_vix_date, "-", "-", "-"]
    })
    st.table(price_levels)

    # Static Line Chart - SPY & VIX Prices
    st.subheader("SPY and VIX Prices Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spy_data.index, spy_data["Close"], label="SPY")
    ax.plot(vix_data.index, vix_data["Close"], label="VIX")
    ax.set_title("SPY and VIX Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Interactive Scatterplot - SPY vs. VIX Daily Returns
    st.subheader("SPY vs. VIX Daily Returns Correlation")
    combined = pd.DataFrame({
        "SPY Returns": spy_data["Returns"],
        "VIX Returns": vix_data["Returns"]
    }).dropna()
    fig = px.scatter(combined, x="SPY Returns", y="VIX Returns", title="SPY vs. VIX Daily Returns Correlation", opacity=0.5)
    st.plotly_chart(fig)

    # Static Normalized Line Chart - SPY & VIX
    st.subheader("Normalized SPY and VIX Levels")
    spy_normalized = (spy_data['Close'] / spy_data['Close'].iloc[0]) * 100
    vix_normalized = (vix_data['Close'] / vix_data['Close'].iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spy_data.index, spy_normalized, label="SPY (Normalized)")
    ax.plot(vix_data.index, vix_normalized, label="VIX (Normalized)")
    ax.set_title("Normalized SPY and VIX Levels Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Level (Starting at 100)")
    ax.legend()
    st.pyplot(fig)

    # Interactive Heatmaps - VIX Spike Analysis
    heatmap_data_2sd = vix_data[vix_data['Spike Level'] == '2SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
    heatmap_data_3sd = vix_data[vix_data['Spike Level'] == '3SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)

    # Heatmap for 2SD Spikes
    st.subheader("Heatmap of 2SD VIX Spikes")
    st.markdown("The **2SD heatmap** visualizes the frequency of VIX spikes ≥ two standard deviations above the mean.")
    fig = px.imshow(
        heatmap_data_2sd,
        labels={"color": "Spike Count"},
        title="Heatmap of 2SD VIX Spikes (Calendar Year)"
    )
    fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig)

    # Heatmap for 3SD Spikes
    st.subheader("Heatmap of 3SD VIX Spikes")
    st.markdown("The **3SD heatmap** visualizes the frequency of VIX spikes ≥ three standard deviations above the mean.")
    fig = px.imshow(
        heatmap_data_3sd,
        labels={"color": "Spike Count"},
        title="Heatmap of 3SD VIX Spikes (Calendar Year)"
    )
    fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig)
