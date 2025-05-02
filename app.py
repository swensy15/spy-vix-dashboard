import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ── 0) Ensure your MD_AUTH_TOKEN is set before importing the SDK ──
# Streamlit Cloud injects any [env] keys in secrets.toml into os.environ
# If you stored it under [env], nothing else is needed; otherwise:
# os.environ["MD_AUTH_TOKEN"] = st.secrets["MD_AUTH_TOKEN"]

# ── 1) Imports ──
import morningstar_data as md
from morningstar_data.direct import InvestmentIdentifier, data_type

st.title("Live SPY and VIX Analysis")

# ── 2) Fetch 10-year daily Close series ──
@st.cache_data(ttl=3600)
def fetch_data():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    # Build the request: PX_LAST is the end-of-day price; isTsdp=True makes it time-series :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    dp = [{"datapointId": "PX_LAST", "isTsdp": True,
           "startDate": start, "endDate": end}]

    inv = [InvestmentIdentifier(ticker="SPY"),
           InvestmentIdentifier(ticker="^VIX")]

    # LONG format: one row per date & ID, so we can pivot it :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
    df = md.direct.get_investment_data(
        investments=inv,
        data_points=dp,
        time_series_format=data_type.TimeSeriesFormat.LONG
    )

    # Pivot into wide form, sort, and rename columns
    df = df.pivot(index="Date", columns="Id", values="PX_LAST")
    df.sort_index(inplace=True)
    df.columns = ["SPY", "VIX"]

    # Build DataFrames matching your old API
    spy = pd.DataFrame({"Close": df["SPY"]})
    vix = pd.DataFrame({"Close": df["VIX"]})
    spy["Returns"] = spy["Close"].pct_change()
    vix["Returns"] = vix["Close"].pct_change()
    return spy, vix

# ── 3) Fetch latest Close (last available day) ──
@st.cache_data(ttl=300)
def fetch_latest():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(days=2)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    dp = [{"datapointId": "PX_LAST", "isTsdp": True,
           "startDate": start, "endDate": end}]

    inv = [InvestmentIdentifier(ticker="SPY"),
           InvestmentIdentifier(ticker="^VIX")]

    df = md.direct.get_investment_data(
        investments=inv,
        data_points=dp,
        time_series_format=data_type.TimeSeriesFormat.LONG
    )
    df = df.pivot(index="Date", columns="Id", values="PX_LAST")
    df.sort_index(inplace=True)
    df.columns = ["SPY", "VIX"]
    return df.tail(1)

# ── 4) Load data ──
spy_data, vix_data = fetch_data()
latest = fetch_latest()

current_spy_price = float(latest["SPY"].iloc[0])
current_spy_date  = latest.index[0].strftime("%Y-%m-%d")
current_vix_price = float(latest["VIX"].iloc[0])
current_vix_date  = latest.index[0].strftime("%Y-%m-%d")

# ── 5) Your existing analysis & plotting ──
threshold_2sd = vix_data["Close"].mean() + 2 * vix_data["Close"].std()
threshold_3sd = vix_data["Close"].mean() + 3 * vix_data["Close"].std()

vix_data["Spike Level"] = np.where(
    vix_data["Close"] >= threshold_3sd, "3SD",
    np.where(vix_data["Close"] >= threshold_2sd, "2SD", "No Spike")
)
vix_data["Month"] = vix_data.index.month
vix_data["Day"  ] = vix_data.index.day

sorted_vix = np.sort(vix_data["Close"])
vix_pct    = (sorted_vix < current_vix_price).sum() / len(sorted_vix) * 100

# Display table
st.subheader("Current SPY & VIX Prices, Threshold Levels, and VIX Percentile Rank")
price_levels = pd.DataFrame({
    "Metric": ["Current SPY Price", "Current VIX Price",
               "2 Standard Deviations (VIX)", "3 Standard Deviations (VIX)",
               "VIX Percentile Rank"],
    "Value": [current_spy_price, current_vix_price,
              threshold_2sd, threshold_3sd,
              f"{vix_pct:.2f}th Percentile"],
    "Date": [current_spy_date, current_vix_date, "-", "-", "-"]
})
st.table(price_levels)

# Static line chart
st.subheader("SPY and VIX Prices Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(spy_data.index, spy_data["Close"], label="SPY")
ax.plot(vix_data.index, vix_data["Close"], label="VIX")
ax.set(title="SPY and VIX Prices Over Time", xlabel="Date", ylabel="Price")
ax.legend()
st.pyplot(fig)

# Scatterplot of returns
st.subheader("SPY vs. VIX Daily Returns Correlation")
combined = pd.DataFrame({
    "SPY Returns": spy_data["Returns"],
    "VIX Returns": vix_data["Returns"]
}).dropna()
fig = px.scatter(combined, x="SPY Returns", y="VIX Returns",
                 title="SPY vs. VIX Daily Returns Correlation", opacity=0.5)
st.plotly_chart(fig)

# Normalized price chart
st.subheader("Normalized SPY and VIX Levels")
spy_norm = (spy_data["Close"] / spy_data["Close"].iloc[0]) * 100
vix_norm = (vix_data["Close"] / vix_data["Close"].iloc[0]) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(spy_data.index, spy_norm, label="SPY (Normalized)")
ax.plot(vix_data.index, vix_norm, label="VIX (Normalized)")
ax.set(title="Normalized SPY and VIX Levels Over Time", xlabel="Date", ylabel="Normalized Level (100 Base)")
ax.legend()
st.pyplot(fig)

# Heatmaps of spikes
heatmap_2sd = vix_data[vix_data["Spike Level"]=="2SD"].groupby(["Month","Day"]).size().unstack(fill_value=0)
heatmap_3sd = vix_data[vix_data["Spike Level"]=="3SD"].groupby(["Month","Day"]).size().unstack(fill_value=0)

st.subheader("Heatmap of 2SD VIX Spikes")
st.markdown("Frequency of VIX ≥ 2σ above mean")
fig = px.imshow(heatmap_2sd, labels={"color":"Spike Count"}, title="2SD VIX Spikes (Calendar Year)")
fig.update_layout(xaxis_title="Day", yaxis_title="Month")
st.plotly_chart(fig)

st.subheader("Heatmap of 3SD VIX Spikes")
st.markdown("Frequency of VIX ≥ 3σ above mean")
fig = px.imshow(heatmap_3sd, labels={"color":"Spike Count"}, title="3SD VIX Spikes (Calendar Year)")
fig.update_layout(xaxis_title="Day", yaxis_title="Month")
st.plotly_chart(fig)
