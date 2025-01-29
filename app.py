import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Only for interactive charts

st.title("Live SPY and VIX Analysis")

# Fetch data for SPY and VIX
@st.cache_data
def fetch_data():
    spy = yf.download("SPY", period="10y", interval="1d")
    vix = yf.download("^VIX", period="10y", interval="1d")
    spy['Returns'] = spy['Close'].pct_change()
    vix['Returns'] = vix['Close'].pct_change()
    return spy, vix

spy_data, vix_data = fetch_data()

# Fetch latest VIX data
latest_vix = yf.Ticker("^VIX").history(period="1d")
current_vix_price = latest_vix['Close'].iloc[-1]  
current_vix_date = latest_vix.index[-1].strftime("%Y-%m-%d")  

if vix_data is not None and not vix_data.empty:
    threshold_2sd = vix_data['Close'].mean() + 2 * vix_data['Close'].std()
    threshold_3sd = vix_data['Close'].mean() + 3 * vix_data['Close'].std()

    vix_data['Spike Level'] = np.where(
        vix_data['Close'] >= threshold_3sd, '3SD',
        np.where(vix_data['Close'] >= threshold_2sd, '2SD', 'No Spike')
    )

    # Extract month and day
    vix_data['Month'] = vix_data.index.month
    vix_data['Day'] = vix_data.index.day

    # Calculate percentile rank
    current_vix_price = float(current_vix_price)  
    sorted_vix_prices = np.sort(vix_data['Close'])  
    vix_percentile = (sorted_vix_prices < current_vix_price).sum() / len(sorted_vix_prices) * 100

    # Display Data Table
    st.subheader("Current VIX Price, Threshold Levels, and Percentile Rank")
    vix_levels = pd.DataFrame({
        "Metric": ["Current VIX Price", "2 Standard Deviations", "3 Standard Deviations", "Percentile Rank"],
        "Value": [
            float(current_vix_price),
            float(threshold_2sd),
            float(threshold_3sd),
            f"{vix_percentile:.2f}th Percentile"
        ],
        "Date": [current_vix_date, "-", "-", "-"]
    })
    st.table(vix_levels)

    # 📈 **Static Line Chart - SPY & VIX Prices**
    st.subheader("SPY and VIX Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spy_data.index, spy_data["Close"], label="SPY")
    ax.plot(vix_data.index, vix_data["Close"], label="VIX")
    ax.set_title("SPY and VIX Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # 📊 **Interactive Scatterplot - SPY vs. VIX Daily Returns**
    st.subheader("SPY vs. VIX Daily Returns Correlation")
    combined = pd.DataFrame({
        "SPY Returns": spy_data["Returns"],
        "VIX Returns": vix_data["Returns"]
    }).dropna()
    fig = px.scatter(combined, x="SPY Returns", y="VIX Returns", title="SPY vs. VIX Daily Returns Correlation", opacity=0.5)
    st.plotly_chart(fig)

    # 📉 **Static Normalized Line Chart - SPY & VIX**
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

    # 🔥 **Interactive Heatmaps - VIX Spike Analysis**
    heatmap_data_2sd = vix_data[vix_data['Spike Level'] == '2SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)
    heatmap_data_3sd = vix_data[vix_data['Spike Level'] == '3SD'].groupby(['Month', 'Day']).size().unstack(fill_value=0)

    # Heatmap for 2SD Spikes
    st.subheader("Heatmap of 2SD VIX Spikes")
    st.markdown("""
    The **2SD heatmap** visualizes the frequency of VIX spikes that were greater than or equal to **two standard deviations above the mean**. 
    """)
    fig = px.imshow(heatmap_data_2sd, labels={"color": "Spike Count"}, title="Heatmap of 2SD VIX Spikes (Calendar Year"))
    st.plotly_chart(fig)

    # Heatmap for 3SD Spikes
    st.subheader("Heatmap of 3SD VIX Spikes")
    st.markdown("""
    The **3SD heatmap** visualizes the frequency of VIX spikes that were greater than or equal to **three standard deviations above the mean**. 
    """)
    fig = px.imshow(heatmap_data_3sd, labels={"color": "Spike Count"}, title="Heatmap of 3SD VIX Spikes (Calendar Year"))
    st.plotly_chart(fig)

else:
    st.error("Failed to fetch or process VIX data.")
