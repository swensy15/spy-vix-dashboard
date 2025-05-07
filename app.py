import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from fredapi import Fred
from polygon import RESTClient

# Streamlit page configuration
st.set_page_config(page_title="SPY & VIX Dashboard", layout="wide")
st.title("Live SPY and VIX Analysis")
st.markdown("**Note**: SPY data is sourced from Polygon.io, and VIX data is sourced from FRED (Federal Reserve Economic Data).")

# Load API keys from Streamlit secrets or environment variables
try:
    POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
except KeyError:
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        st.error("Polygon API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        st.error("FRED API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

# Initialize Polygon client
client = RESTClient(POLYGON_API_KEY)

# Cache data fetching to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_spy_data(symbol):
    try:
        aggs = []
        for a in client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_="2023-01-01",  # Adjust date range as needed
            to="2025-05-07",    # Current date
            limit=50000
        ):
            aggs.append(a)
        
        if not aggs:
            st.error(f"Error fetching data for {symbol}: No data returned")
            return None
            
        df = pd.DataFrame(aggs)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp")
        
        # Rename columns
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        df.set_index("timestamp", inplace=True)
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_vix_data(fred_api_key):
    try:
        fred = Fred(api_key=fred_api_key)
        vix_series = fred.get_series('VIXCLS')
        df = pd.DataFrame(vix_series, columns=['Close'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change(fill_method=None)
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch VIX data from FRED: {str(e)}")
        return None

# Fetch latest prices
def fetch_latest_spy_price(symbol):
    try:
        trade = client.get_last_trade(symbol)
        if not trade or "price" not in trade:
            st.error(f"Failed to fetch latest price for {symbol}: No data returned")
            return None, None
        price = trade["price"]
        date = pd.to_datetime(trade["timestamp"], unit="ms").strftime("%Y-%m-%d")
        return price, date
    except Exception as e:
        st.error(f"Failed to fetch latest price for {symbol}: {str(e)}")
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
        return None, None

# Load SPY and VIX data
def load_data(polygon_key, fred_key):
    spy_data = fetch_spy_data("SPY")
    vix_data = fetch_vix_data(fred_key)
    return spy_data, vix_data

# Main app logic
if POLYGON_API_KEY and FRED_API_KEY:
    # Load historical data
    spy_data, vix_data = load_data(POLYGON_API_KEY, FRED_API_KEY)
    
    # Load latest prices
    current_spy_price, current_spy_date = fetch_latest_spy_price("SPY")
    current_vix_price, current_vix_date = fetch_latest_vix_price(FRED_API_KEY)
    
    if spy_data is not None and vix_data is not None and not spy_data.empty and not vix_data.empty:
        # Create a copy of vix_data for heatmap purposes
        vix_data_full = vix_data.copy()
        
        # Align SPY and VIX data for other plots
        aligned_dates = spy_data.index.intersection(vix_data.index)
        spy_data_aligned = spy_data.loc[aligned_dates]
        vix_data_aligned = vix_data.loc[aligned_dates]
        
        # Calculate VIX thresholds using the full dataset
        threshold_2sd = vix_data_full['Close'].mean() + 2 * vix_data_full['Close'].std()
        threshold_3sd = vix_data_full['Close'].mean() + 3 * vix_data_full['Close'].std()
        
        # Assign spike levels using the full dataset
        vix_data_full['Spike Level'] = np.where(
            vix_data_full['Close'] >= threshold_3sd, '3SD',
            np.where(vix_data_full['Close'] >= threshold_2sd, '2SD', 'No Spike')
        )
        
        # Extract month and day for heatmap
        vix_data_full['Month'] = vix_data_full.index.month
        vix_data_full['Day'] = vix_data_full.index.day
        
        # Use aligned data for other calculations
        vix_data = vix_data_aligned
        spy_data = spy_data_aligned
        
        # Calculate VIX percentile rank
        sorted_vix_prices = np.sort(vix_data_full['Close'])
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
        heatmap_data_2sd = vix_data_full[vix_data_full['Spike Level'] == '2SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
        heatmap_data_2sd = heatmap_data_2sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
        fig = px.imshow(
            heatmap_data_2sd,
            labels={"color": "Spike Count"},
            title=f"Heatmap of 2SD VIX Spikes (Historical, {vix_data_full.index.year.min()}-{vix_data_full.index.year.max()})",
            color_continuous_scale="YlOrRd"
        )
        fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
        st.plotly_chart(fig)

        st.subheader("Heatmap of 3SD VIX Spikes")
        st.markdown("Visualizes the frequency of VIX spikes ≥ 3 standard deviations above the mean across all historical data.")
        heatmap_data_3sd = vix_data_full[vix_data_full['Spike Level'] == '3SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
        heatmap_data_3sd = heatmap_data_3sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
        fig = px.imshow(
            heatmap_data_3sd,
            labels={"color": "Spike Count"},
            title=f"Heatmap of 3SD VIX Spikes (Historical, {vix_data_full.index.year.min()}-{vix_data_full.index.year.max()})",
            color_continuous_scale="YlOrRd"
        )
        fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
        st.plotly_chart(fig)
    
    else:
        st.error("Failed to load SPY or VIX data. Please check your API keys or try again later.")
