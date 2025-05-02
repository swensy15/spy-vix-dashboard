import base64
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Live SPY and VIX Analysis")

# --- 1) AUTH: fetch & cache an OAuth token ---
@st.cache_data(ttl=55*60)
def get_ms_token():
    creds = f"{st.secrets.morningstar.username}:{st.secrets.morningstar.password}"
    b64 = base64.b64encode(creds.encode()).decode()
    resp = requests.post(
        "https://www.us-api.morningstar.com/token/oauth",
        headers={"Authorization": f"Basic {b64}"}
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

# --- 2) Helper to pull time series from Morningstar ---
@st.cache_data(ttl=3600)
def fetch_ms_timeseries(identifier: str, start: str, end: str, freq: str = "daily"):
    token = get_ms_token()
    r = requests.get(
        "https://www.us-api.morningstar.com/direct-web-services/v1/time-series",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "identifiers": identifier,
            "startDate": start,
            "endDate": end,
            "frequency": freq
        }
    )
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df

# --- 3) Replace your old fetch_data() ---
@st.cache_data
def fetch_data():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    spy = fetch_ms_timeseries("SPY", start, end)
    vix = fetch_ms_timeseries("^VIX", start, end)

    # align column names with your old code
    spy["Close"] = spy["close"]
    vix["Close"] = vix["close"]

    spy["Returns"] = spy["Close"].pct_change()
    vix["Returns"] = vix["Close"].pct_change()
    return spy, vix

# --- 4) Replace your latest-data logic ---
def fetch_latest_point(idstr):
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(days=2)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    df   = fetch_ms_timeseries(idstr, start, end)
    df["Close"] = df["close"]
    return df.tail(1)

spy_data, vix_data = fetch_data()
latest_spy = fetch_latest_point("SPY")
latest_vix = fetch_latest_point("^VIX")

# pull the single most recent row
current_spy_price = latest_spy["Close"].iloc[-1]
current_spy_date  = latest_spy.index[-1].strftime("%Y-%m-%d")
current_vix_price = latest_vix["Close"].iloc[-1]
current_vix_date  = latest_vix.index[-1].strftime("%Y-%m-%d")

# --- 5) Your existing analytics & charts follow exactly as before ---
if not vix_data.empty:
    threshold_2sd = vix_data['Close'].mean() + 2 * vix_data['Close'].std()
    threshold_3sd = vix_data['Close'].mean() + 3 * vix_data['Close'].std()

    vix_data['Spike Level'] = np.where(
        vix_data['Close'] >= threshold_3sd, '3SD',
        np.where(vix_data['Close'] >= threshold_2sd, '2SD', 'No Spike')
    )
    vix_data['Month'] = vix_data.index.month
    vix_data['Day']   = vix_data.index.day

    sorted_vix = np.sort(vix_data['Close'])
    vix_pct   = (sorted_vix < current_vix_price).sum() / len(sorted_vix) * 100

    st.subheader("Current SPY & VIX Prices, Threshold Levels, and VIX Percentile Rank")
    price_levels = pd.DataFrame({
        "Metric": ["Current SPY Price", "Current VIX Price", "2 Standard Deviations (VIX)", 
                   "3 Standard Deviations (VIX)", "VIX Percentile Rank"],
        "Value": [
            float(current_spy_price),
            float(current_vix_price),
            float(threshold_2sd),
            float(threshold_3sd),
            f"{vix_pct:.2f}th Percentile"
        ],
        "Date": [current_spy_date, current_vix_date, "-", "-", "-"]
    })
    st.table(price_levels)

    # … then all your plotting code exactly as before …
    # (static line, scatter, normalized line, heatmaps)
else:
    st.error("Failed to fetch or process VIX data.")
