import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fredapi import Fred
import requests
from bs4 import BeautifulSoup
import os

# Streamlit page configuration
st.set_page_config(page_title="VIX Dashboard", layout="wide")
st.title("VIX Analysis")
st.markdown("**Note**: VIX historical data and S&P 500 data are sourced from FRED (Federal Reserve Economic Data). VIX current price and SPY price are sourced from a delayed public web feed.")

# Load API key from Streamlit secrets or environment variables
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        st.error("FRED API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()

# Cache data fetching to improve performance
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

# Fetch S&P 500 data from FRED
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sp500_data(fred_api_key):
    try:
        fred = Fred(api_key=fred_api_key)
        sp500_series = fred.get_series('SP500', observation_start='1990-01-01')
        df = pd.DataFrame(sp500_series, columns=['SP500'])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 data from FRED: {str(e)}")
        return None

# Fetch latest VIX price from Google Finance
@st.cache_data(ttl=300)  # Cache for 5 minutes due to delayed data
def fetch_latest_vix_price():
    try:
        url = "https://www.google.com/finance/quote/VIX:INDEXCBOE?hl=en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        price_element = soup.find('div', class_='YMlKec fxKbKc')
        if price_element:
            price = float(price_element.text.replace(',', ''))
            date_element = soup.find('div', class_='ygUjEc')  # Approximate class for date
            date = date_element.text if date_element else "Delayed"
            return price, date
        else:
            st.error("Failed to parse VIX price from web source.")
            return None, None
    except Exception as e:
        st.error(f"Failed to fetch VIX price: {str(e)}")
        return None, None

# Fetch delayed SPY price from Google Finance
@st.cache_data(ttl=300)  # Cache for 5 minutes due to delayed data
def fetch_spy_price():
    try:
        url = "https://www.google.com/finance/quote/SPY:NYSEARCA"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        price_element = soup.find('div', class_='YMlKec fxKbKc')
        if price_element:
            price = float(price_element.text.replace('$', '').replace(',', ''))
            date_element = soup.find('div', class_='ygUjEc')  # Approximate class for date
            date = date_element.text if date_element else "Delayed"
            return price, date
        else:
            st.error("Failed to parse SPY price from web source.")
            return None, None
    except Exception as e:
        st.error(f"Failed to fetch SPY price: {str(e)}")
        return None, None

# Load VIX and S&P 500 data
def load_data(fred_key):
    vix_data = fetch_vix_data(fred_key)
    sp500_data = fetch_sp500_data(fred_key)
    if vix_data is not None and sp500_data is not None:
        # Merge VIX and S&P 500 data on the date index
        combined_data = vix_data.join(sp500_data, how='inner')
        return vix_data, combined_data
    return None, None

# Main app logic
if FRED_API_KEY:
    # Load historical data
    vix_data, combined_data = load_data(FRED_API_KEY)
    
    # Load latest prices
    current_vix_price, current_vix_date = fetch_latest_vix_price()
    current_spy_price, current_spy_date = fetch_spy_price()
    
    if vix_data is not None and not vix_data.empty and combined_data is not None and not combined_data.empty:
        # Create a copy of vix_data for heatmap purposes
        vix_data_full = vix_data.copy()
        
        # Calculate VIX thresholds using the full dataset
        threshold_2sd = vix_data_full['Close'].mean() + 2 * vix_data_full['Close'].std()
        threshold_3sd = vix_data_full['Close'].mean() + 3 * vix_data_full['Close'].std()
        
        # Calculate VIX median and mean
        vix_median = vix_data_full['Close'].median()
        vix_mean = vix_data_full['Close'].mean()
        
        # Assign spike levels using the full dataset
        vix_data_full['Spike Level'] = np.where(
            vix_data_full['Close'] >= threshold_3sd, '3SD',
            np.where(vix_data_full['Close'] >= threshold_2sd, '2SD', 'No Spike')
        )
        
        # Extract month and day for heatmap
        vix_data_full['Month'] = vix_data_full.index.month
        vix_data_full['Day'] = vix_data_full.index.day
        
        # Calculate VIX percentile rank
        sorted_vix_prices = np.sort(vix_data_full['Close'])
        vix_percentile = (sorted_vix_prices < current_vix_price).sum() / len(sorted_vix_prices) * 100 if current_vix_price else 0
        
        # Display current prices, thresholds, and statistics
        st.subheader("Current VIX & SPY Prices, Thresholds, and VIX Statistics")
        price_levels = pd.DataFrame({
            "Metric": ["Current VIX Price", "Current SPY Price", "2 Standard Deviations (VIX)", "3 Standard Deviations (VIX)", "VIX Percentile Rank", "VIX Median", "VIX Mean"],
            "Value": [
                f"{current_vix_price:.2f}" if current_vix_price else "N/A",
                f"{current_spy_price:.2f}" if current_spy_price else "N/A",
                f"{threshold_2sd:.2f}",
                f"{threshold_3sd:.2f}",
                f"{vix_percentile:.2f}th Percentile" if current_vix_price else "N/A",
                f"{vix_median:.2f}",
                f"{vix_mean:.2f}"
            ],
            "Date": [
                current_vix_date if current_vix_date else "N/A",
                current_spy_date if current_spy_date else "N/A",
                "-", "-", "-", "-", "-"
            ]
        })
        st.table(price_levels)
        
        # Line Chart: VIX and S&P 500 Prices Over Time
        st.subheader("VIX and S&P 500 Prices Over Time")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot VIX on the left y-axis
        ax1.plot(combined_data.index, combined_data["Close"], label="VIX", color="orange")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("VIX Price", color="orange")
        ax1.tick_params(axis='y', labelcolor="orange")
        ax1.grid(True)
        
        # Create a second y-axis for S&P 500
        ax2 = ax1.twinx()
        ax2.plot(combined_data.index, combined_data["SP500"], label="S&P 500", color="blue")
        ax2.set_ylabel("S&P 500 Price", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")
        
        # Add title and legend
        fig.suptitle("VIX and S&P 500 Prices Over Time")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
        
        st.pyplot(fig)
        
       # Heatmaps for VIX Spikes
st.subheader("Heatmap of 2SD VIX Spikes")
st.markdown("Visualizes the frequency of VIX spikes ≥ 2 standard deviations above the mean across all historical data.")
# Filter 2SD spikes and group by Month, Day, and collect years
spikes_2sd = vix_data_full[vix_data_full['Spike Level'] == '2SD'].copy()
spikes_2sd['Year'] = spikes_2sd.index.year
heatmap_data_2sd = spikes_2sd.groupby(['Month', 'Day']).size().unstack(fill_value=0)
heatmap_data_2sd = heatmap_data_2sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
# Create a DataFrame to store years for each (Month, Day)
years_2sd = spikes_2sd.groupby(['Month', 'Day'])['Year'].apply(lambda x: ', '.join(x.astype(str))).unstack(fill_value='')
years_2sd = years_2sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value='')
# Create custom hover text
hover_text_2sd = [[f"Month: {month}<br>Day: {day}<br>Spike Count: {heatmap_data_2sd.loc[month, day]}<br>Years: {years_2sd.loc[month, day]}" 
                   for day in range(1, 32)] for month in range(1, 13)]
fig = px.imshow(
    heatmap_data_2sd,
    labels={"color": "Spike Count"},
    title=f"Heatmap of 2SD VIX Spikes (Historical, {vix_data_full.index.year.min()}-{vix_data_full.index.year.max()})",
    color_continuous_scale="YlOrRd",
    text_auto=False
)
fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
fig.update_traces(hovertemplate='%{text}', text=hover_text_2sd)
st.plotly_chart(fig)

st.subheader("Heatmap of 3SD VIX Spikes")
st.markdown("Visualizes the frequency of VIX spikes ≥ 3 standard deviations above the mean across all historical data.")
# Filter 3SD spikes and group by Month, Day, and collect years
spikes_3sd = vix_data_full[vix_data_full['Spike Level'] == '3SD'].copy()
spikes_3sd['Year'] = spikes_3sd.index.year
heatmap_data_3sd = spikes_3sd.groupby(['Month', 'Day']).size().unstack(fill_value=0)
heatmap_data_3sd = heatmap_data_3sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value=0)
# Create a DataFrame to store years for each (Month, Day)
years_3sd = spikes_3sd.groupby(['Month', 'Day'])['Year'].apply(lambda x: ', '.join(x.astype(str))).unstack(fill_value='')
years_3sd = years_3sd.reindex(index=range(1, 13), columns=range(1, 32), fill_value='')
# Create custom hover text
hover_text_3sd = [[f"Month: {month}<br>Day: {day}<br>Spike Count: {heatmap_data_3sd.loc[month, day]}<br>Years: {years_3sd.loc[month, day]}" 
                   for day in range(1, 32)] for month in range(1, 13)]
fig = px.imshow(
    heatmap_data_3sd,
    labels={"color": "Spike Count"},
    title=f"Heatmap of 3SD VIX Spikes (Historical, {vix_data_full.index.year.min()}-{vix_data_full.index.year.max()})",
    color_continuous_scale="YlOrRd",
    text_auto=False
)
fig.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
fig.update_traces(hovertemplate='%{text}', text=hover_text_3sd)
st.plotly_chart(fig)
