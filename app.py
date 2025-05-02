# app.py

import base64
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os

st.title("Live SPY and VIX Analysis")

# ── 1) Load your Morningstar credentials securely ──
# Put these in ~/.streamlit/secrets.toml under [morningstar], or in Streamlit Cloud Secrets
# [morningstar]
username = st.secrets.morningstar.username
password = st.secrets.morningstar.password

# ── 2) Fetch & cache an OAuth token ──
@st.cache_data(ttl=55*60)
def get_ms_token():
    creds = f"{username}:{password}"
    b64 = base64.b64encode(creds.encode()).decode()
    url = "https://www.us-api.morningstar.com/token/oauth"
    r = requests.post(url, headers={"Authorization": f"Basic {b64}"})
    r.raise_for_status()
    return r.json()["access_token"]

# ── 3) Time‐series fetcher ──
def fetch_ms_timeseries(ticker: str, start: str, end: str, frequency: str = "daily"):
    """
    Returns a DataFrame indexed by date, with a 'close' column.
    """
    token = get_ms_token()
    resp = requests.get(
        "https://www.us-api.morningstar.com/direct-web-services/v1/time-series",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "identifiers": ticker,
            "startDate": start,
            "endDate": end,
            "frequency": frequency
        }
    )
    resp.raise_for_status()
    js = resp.json()
    df = pd.DataFrame(js["data"])        # each item has 'date' & 'close'
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df

# ── 4) Replace fetch_data() ──
@st.cache_data(ttl=3600)
def fetch_data():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    spy = fetch_ms_timeseries("SPY", start, end)
    vix = fetch_ms_timeseries("^VIX", start, end)

    # match your old schema
    spy["Close"]   = spy["close"]
    spy["Returns"] = spy["close"].pct_change()
    vix["Close"]   = vix["close"]
    vix["Returns"] = vix["close"].pct_change()
    return spy, vix

# ── 5) Replace latest‐data logic ──
@st.cache_data(ttl=300)
def fetch_latest():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(days=2)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    spy = fetch_ms_timeseries("SPY", start, end)
    vix = fetch_ms_timeseries("^VIX", start, end)

    return spy.tail(1), vix.tail(1)

# ── 6) Load everything ──
spy_data, vix_data = fetch_data()
latest_spy, latest_vix = fetch_latest()

if spy_data.empty or vix_data.empty or latest_spy.empty or latest_vix.empty:
    st.error("Failed to fetch data. Check your token, credentials, and network.")
    st.stop()

current_spy_price = float(latest_spy["Close"])
current_spy_date  = latest_spy.index[0].strftime("%Y-%m-%d")
current_vix_price = float(latest_vix["Close"])
current_vix_date  = latest_vix.index[0].strftime("%Y-%m-%d")

# ── 7) Analytics & plots (unchanged) ──
threshold_2sd = vix_data["Close"].mean() + 2 * vix_data["Close"].std()
threshold_3sd = vix_data["Close"].mean() + 3 * vix_data["Close"].std()

vix_data["Spike Level"] = np.where(
    vix_data["Close"] >= threshold_3sd, "3SD",
    np.where(vix_data["Close"] >= threshold_2sd, "2SD", "No Spike")
)
vix_data["Month"] = vix_data.index.month
vix_data["Day"]   = vix_data.index.day

sorted_vix = np.sort(vix_data["Close"])
vix_pct    = (sorted_vix < current_vix_price).sum() / len(sorted_vix) * 100

st.subheader("Current SPY & VIX Prices, Threshold Levels, and VIX Percentile Rank")
price_levels = pd.DataFrame({
    "Metric": [
        "Current SPY Price","Current VIX Price",
        "2 Standard Deviations (VIX)","3 Standard Deviations (VIX)",
        "VIX Percentile Rank"
    ],
    "Value": [
        current_spy_price, current_vix_price,
        threshold_2sd, threshold_3sd,
        f"{vix_pct:.2f}th Percentile"
    ],
    "Date": [current_spy_date, current_vix_date, "-", "-", "-"]
})
st.table(price_levels)

st.subheader("SPY and VIX Prices Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(spy_data.index, spy_data["Close"], label="SPY")
ax.plot(vix_data.index, vix_data["Close"], label="VIX")
ax.set(title="SPY and VIX Prices Over Time", xlabel="Date", ylabel="Price")
ax.legend()
st.pyplot(fig)

st.subheader("SPY vs. VIX Daily Returns Correlation")
combined = pd.DataFrame({
    "SPY Returns": spy_data["Returns"],
    "VIX Returns": vix_data["Returns"]
}).dropna()
fig = px.scatter(combined, x="SPY Returns", y="VIX Returns", opacity=0.5,
                 title="SPY vs. VIX Daily Returns Correlation")
st.plotly_chart(fig)

st.subheader("Normalized SPY and VIX Levels")
spy_norm = (spy_data["Close"] / spy_data["Close"].iloc[0]) * 100
vix_norm = (vix_data["Close"] / vix_data["Close"].iloc[0]) * 100
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(spy_data.index, spy_norm, label="SPY (Normalized)")
ax.plot(vix_data.index, vix_norm, label="VIX (Normalized)")
ax.set(title="Normalized SPY and VIX Levels Over Time", xlabel="Date", ylabel="Normalized Level (100 Base)")
ax.legend()
st.pyplot(fig)

heatmap_2sd = vix_data[vix_data["Spike Level"]=="2SD"].groupby(["Month","Day"]).size().unstack(fill_value=0)
heatmap_3sd = vix_data[vix_data["Spike Level"]=="3SD"].groupby(["Month","Day"]).size().unstack(fill_value=0)

st.subheader("Heatmap of 2SD VIX Spikes")
st.markdown("Frequency of VIX ≥ 2σ above mean")
fig = px.imshow(heatmap_2sd, labels={"color":"Spike Count"}, title="Heatmap of 2SD VIX Spikes")
fig.update_layout(xaxis_title="Day", yaxis_title="Month")
st.plotly_chart(fig)

st.subheader("Heatmap of 3SD VIX Spikes")
st.markdown("Frequency of VIX ≥ 3σ above mean")
fig = px.imshow(heatmap_3sd, labels={"color":"Spike Count"}, title="Heatmap of 3SD VIX Spikes")
fig.update_layout(xaxis_title="Day", yaxis_title="Month")
st.plotly_chart(fig)
