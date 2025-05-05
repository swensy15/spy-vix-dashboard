import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import os

# Streamlit page configuration
st.set_page_config(page_title="SPY & VIX Dashboard", layout="wide")
st.title("Live SPY and VIX Analysis")

# Load Alpha Vantage API key from Streamlit secrets or environment variable
try:
    API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
except KeyError:
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not API_KEY:
        st.error("Alpha Vantage API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

# Alpha Vantage API configuration
BASE_URL = "https://www.alphavantage.co/query"

# Cache data fetching to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(symbol, api_key):
    try:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",  # Fetch full history (up to 20 years)
            "apikey": api_key,
            "datatype": "json"
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            st.error(f"Error fetching data for {symbol}: {data.get('Note', 'Unknown error')}")
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
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {str(e)}")
        return None

# Fetch latest prices
def fetch_latest_price(symbol, api_key):
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data:
            return None, None
            
        quote = data["Global Quote"]
        price = float(quote["05. price"])
        date = quote["07. latest trading day"]
        return price, date
    except Exception as e:
        st.error(f"Failed to fetch latest price for {symbol}: {str(e)}")
        return None, None

# Load SPY and VIX data
def load_data(api_key):
    spy_data = fetch_data("SPY", api_key)
    vix_data = fetch_data("VIX", api_key)
    return spy_data, vix_data

# Main app logic
if API_KEY:
    # Load historical data
    spy_data, vix_data = load_data(API_KEY)
    
    # Load latest prices
    current_spy_price, current_spy_date = fetch_latest_price("SPY", API_KEY)
    current_vix_price, current_vix_date = fetch_latest_price("VIX", API_KEY)
    
    if spy_data is not None and vix_data is not None and not spy_data.empty and not vix_data.empty:
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
        heatmap_data_2sd = vix_data[vix_data['Spike Level'] == '2SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
        heatmap_data_3sd = vix_data[vix_data['Spike Level'] == '3SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
        
        # 2SD Heatmap
        st.subheader("Heatmap of 2SD VIX Spikes")
        st.markdown("Visualizes the frequency of VIX spikes ≥ 2 standard deviations above the mean.")
        fig = px.imshow(
            heatmap_data_2sd,
            labels={"color": "Spike Count"},
            title="Heatmap of 2SD VIX Spikes (Calendar Year)",
            color_continuous_scale="YlOrRd"
        )
        fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
        st.plotly_chart(fig)
        
        # 3SD Heatmap
        st.subheader("Heatmap of 3SD VIX Spikes")
        st.markdown("Visualizes the frequency of VIX spikes ≥ 3 standard deviations above the mean.")
        fig = px.imshow(
            heatmap_data_3sd,
            labels={"color": "Spike Count"},
            title="Heatmap of 3SD VIX Spikes (Calendar Year)",
            color_continuous_scale="YlOrRd"
        )
        fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
        st.plotly_chart(fig)
    
    else:
        st.error("Failed to load SPY or VIX data. Please check your API key or try again later.")
