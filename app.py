import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import os
from fredapi import Fred

# Streamlit page configuration
st.set_page_config(page_title="SPY & VIX Dashboard", layout="wide")
st.title("Live SPY and VIX Analysis")
st.markdown("**Note**: SPY data is sourced from Alpha Vantage, and VIX data is sourced from FRED (Federal Reserve Economic Data).")

# Load API keys from Streamlit secrets or environment variables
try:
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except KeyError:
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not ALPHA_VANTAGE_API_KEY:
        st.error("Alpha Vantage API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        st.error("FRED API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

# Alpha Vantage API configuration
BASE_URL = "https://www.alphavantage.co/query"

# Cache data fetching to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_spy_data(symbol, api_key):
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",  # Changed to "compact" to reduce API calls
            "apikey": api_key,
            "datatype": "json"
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            error_msg = data.get('Note', 'Unknown error')
            if "minute" in error_msg.lower():
                st.error(f"API rate limit exceeded for {symbol}. Please wait an hour or upgrade your plan.")
            else:
                st.error(f"Error fetching data for {symbol}: {error_msg}")
            st.write(f"Debug: API response for {symbol} - {data}")  # Changed to st.write
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns to match yfinance format
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        
        # Calculate daily returns with fill_method=None to fix warning
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {str(e)}")
        st.write(f"Debug: Fetch error for {symbol} - {str(e)}")  # Changed to st.write
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_vix_data(fred_api_key):
    try:
        fred = Fred(api_key=fred_api_key)
        vix_series = fred.get_series('VIXCLS')  # VIX Close series from FRED
        df = pd.DataFrame(vix_series, columns=['Close'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Calculate daily returns with fill_method=None to fix warning
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch VIX data from FRED: {str(e)}")
        st.write(f"Debug: Fetch VIX error - {str(e)}")  # Changed to st.write
        return None

# Fetch latest prices
def fetch_latest_spy_price(symbol, api_key):
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data or not data["Global Quote"]:
            error_msg = data.get('Note', 'No data returned')
            st.error(f"Failed to fetch latest price for {symbol}: {error_msg}")
            st.write(f"Debug: Latest price API response for {symbol} - {data}")  # Changed to st.write
            return None, None
            
        quote = data["Global Quote"]
        price = float(quote["05. price"])
        date = quote["07. latest trading day"]
        return price, date
    except Exception as e:
        st.error(f"Failed to fetch latest price for {symbol}: {str(e)}")
        st.write(f"Debug: Latest price fetch error for {symbol} - {str(e)}")  # Changed to st.write
        return None, None

@st.cache_data(ttl=3600)
def fetch_latest_vix_price(fred_api_key):
    try:
        fred = Fred(api_key=fred_api_key)
        vix_series = fred.get_series('VIXCLS')
        latest_price = vix_series.iloc[-1]
        latest_date = vix_series.index[-1].strftime("%Y-%m-%d")
        return latest_price, latest_date
    except Exception as e:
        st.error(f"Failed to fetch latest VIX price from FRED: {str(e)}")
        st.write(f"Debug: Latest VIX price fetch error - {str(e)}")  # Changed to st.write
        return None, None

# Load SPY and VIX data
def load_data(alpha_vantage_key, fred_key):
    spy_data = fetch_spy_data("SPY", alpha_vantage_key)
    vix_data = fetch_vix_data(fred_key)
    return spy_data, vix_data

# Main app logic
if ALPHA_VANTAGE_API_KEY and FRED_API_KEY:
    # Load historical data
    spy_data, vix_data = load_data(ALPHA_VANTAGE_API_KEY, FRED_API_KEY)
    
    # Load latest prices
    current_spy_price, current_spy_date = fetch_latest_spy_price("SPY", ALPHA_VANTAGE_API_KEY)
    current_vix_price, current_vix_date = fetch_latest_vix_price(FRED_API_KEY)
    
    if spy_data is not None and vix_data is not None and not spy_data.empty and not vix_data.empty:
        # Align SPY and VIX data (since FRED VIX data might have different dates)
        spy_data = spy_data[spy_data.index.isin(vix_data.index)]
        vix_data = vix_data[vix_data.index.isin(spy_data.index)]
        
        # Calculate VIX thresholds
        threshold_2sd = vix_data['Close'].mean() + 2 * vix_data['Close'].std()
        threshold_3sd = vix_data['Close'].mean() + 3 * vix_data['Close'].std()
        
        # Assign spike levels
        vix_data['Spike Level'] = np.where(
            vix_data['Close'] >= threshold_3sd, '3SD',
            np.where(vix_data['Close'] >= threshold_2sd, '2SD', 'No Spike')
        )
        
        # Extract month and day for heatmap
        vix_data['Month'] = vix_data.index.month
        vix_data['Day'] = vix_data.index.day
        
        # Calculate VIX percentile rank
        sorted_vix_prices = np.sort(vix_data['Close'])
        vix_percentile = (sorted_vix_prices < current_vix_price).sum() / len(sorted_vix_prices) * 100 if current_vix_price else 0
        
        # Display current prices and thresholds
        st.subheader("Current SPY & VIX Prices, Thresholds, and VIX Percentile Rank")
        price_levels = pd.DataFrame({
            "Metric": ["Current SPY Price", "Current VIX Price", "2 Standard Deviations (VIX)", "3 Standard Deviations (VIX)", "VIX Percentile Rank"],
            "Value": [
                f"{current_spy_price:.2f}" if current_spy_price else "N/A",
                f"{current_vix_price:.2f}" if current_vix_price else "N/A",
                f"{threshold_2sd:.2f}",
                f"{threshold_3sd:.2f}",
                f"{vix_percentile:.2f}th Percentile" if current_vix_price else "N/A"
            ],
            "Date": [
                current_spy_date if current_spy_date else "N/A",
                current_vix_date if current_vix_date else "N/A",
                "-", "-", "-"
            ]
        })
        st.table(price_levels)
        
        # Line Chart: SPY & VIX Prices
        st.subheader("SPY and VIX Prices Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(spy_data.index, spy_data["Close"], label="SPY", color="blue")
        ax.plot(vix_data.index, vix_data["Close"], label="VIX", color="orange")
        ax.set_title("SPY and VIX Prices Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Scatterplot: SPY vs. VIX Daily Returns
        st.subheader("SPY vs. VIX Daily Returns Correlation")
        combined = pd.DataFrame({
            "SPY Returns": spy_data["Returns"],
            "VIX Returns": vix_data["Returns"]
        }).dropna()
        fig = px.scatter(
            combined, 
            x="SPY Returns", 
            y="VIX Returns", 
            title="SPY vs. VIX Daily Returns Correlation", 
            opacity=0.5,
            trendline="ols"
        )
        st.plotly_chart(fig)
        
        # Normalized Line Chart
        st.subheader("Normalized SPY and VIX Levels")
        spy_normalized = (spy_data['Close'] / spy_data['Close'].iloc[0]) * 100
        vix_normalized = (vix_data['Close'] / vix_data['Close'].iloc[0]) * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(spy_data.index, spy_normalized, label="SPY (Normalized)", color="blue")
        ax.plot(vix_data.index, vix_normalized, label="VIX (Normalized)", color="orange")
        ax.set_title("Normalized SPY and VIX Levels Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Level (Starting at 100)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
# Heatmaps for VIX Spikes
st.subheader("Heatmap of 2SD VIX Spikes")
st.markdown("Visualizes the frequency of VIX spikes ≥ 2 standard deviations above the mean across all historical data.")
# Create heatmap data for 2SD spikes
heatmap_data_2sd = vix_data[vix_data['Spike Level'] == '2SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
# Ensure all days (1-31) and months (1-12) are included, even with zero counts
heatmap_data_2sd = heatmap_data_2sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
fig = px.imshow(
    heatmap_data_2sd,
    labels={"color": "Spike Count"},
    title=f"Heatmap of 2SD VIX Spikes (Historical, {vix_data.index.year.min()}-{vix_data.index.year.max()})",
    color_continuous_scale="YlOrRd"
)
fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
st.plotly_chart(fig)

st.subheader("Heatmap of 3SD VIX Spikes")
st.markdown("Visualizes the frequency of VIX spikes ≥ 3 standard deviations above the mean across all historical data.")
# Create heatmap data for 3SD spikes
heatmap_data_3sd = vix_data[vix_data['Spike Level'] == '3SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
# Ensure all days (1-31) and months (1-12) are included, even with zero counts
heatmap_data_3sd = heatmap_data_3sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
fig = px.imshow(
    heatmap_data_3sd,
    labels={"color": "Spike Count"},
    title=f"Heatmap of 3SD VIX Spikes (Historical, {vix_data.index.year.min()}-{vix_data.index.year.max()})",
    color_continuous_scale="YlOrRd"
)
fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
st.plotly_chart(fig)
