import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import yfinance as yf
import requests
from io import StringIO

# --- Configuration ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HIST_WINDOW = "10y"
INTERVAL = "1d"

# --- Helpers ---
@st.cache_data
def load_historical_spy() -> pd.DataFrame:
    """
    Load 10-year SPY history from Parquet if present;
    otherwise fetch via yfinance and save.
    """
    fp = DATA_DIR / f"SPY_{HIST_WINDOW}.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
    else:
        df = yf.download("SPY", period=HIST_WINDOW, interval=INTERVAL)
        df.to_parquet(fp)
    df["Returns"] = df["Close"].pct_change()
    return df

@st.cache_data
def load_historical_vix() -> pd.DataFrame:
    """
    Load full VIX history via Stooq CSV (no rate limits).
    """
    url = "https://stooq.com/q/d/l/?s=vix&i=d"
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"]).rename(
        columns=lambda c: c.capitalize()
    ).set_index("Date")
    df["Returns"] = df["Close"].pct_change()
    return df

@st.cache_data(ttl=300)
def fetch_latest_spy() -> (float, pd.Timestamp):
    """
    Fetch the most recent EOD SPY close via Stooq CSV.
    Returns (price, as_of_date).
    """
    url = "https://stooq.com/q/l/?s=spy.us&f=sd2t2ohlc&h&e=csv"
    resp = requests.get(url)
    resp.raise_for_status()
    temp = pd.read_csv(StringIO(resp.text), parse_dates=["Date"])
    last = temp.iloc[-1]
    return float(last["Close"]), last["Date"]

# --- Load data ---
spy = load_historical_spy()
vix = load_historical_vix()

current_spy_price, as_of_dt = fetch_latest_spy()
current_vix_price = float(vix["Close"].iloc[-1])
as_of_date = pd.to_datetime(as_of_dt).strftime("%Y-%m-%d")

# --- Metrics ---
mu, sigma = vix["Close"].agg(["mean", "std"])
th2, th3 = mu + 2 * sigma, mu + 3 * sigma
vix_pct = vix["Close"].rank(pct=True).iloc[-1] * 100

vix = vix.assign(
    Spike=lambda d: np.where(
        d["Close"] >= th3, "3SD",
        np.where(d["Close"] >= th2, "2SD", "No Spike")
    )
).assign(Month=vix.index.month, Day=vix.index.day)

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("📊 SPY & VIX Dashboard")

# Overview table
summary = pd.DataFrame({
    "Metric": ["SPY (Last)", "VIX (Last)", "VIX 2σ", "VIX 3σ", "VIX Percentile"],
    "Value": [
        current_spy_price,
        current_vix_price,
        float(th2),
        float(th3),
        f"{vix_pct:.1f}th"
    ],
    "As Of": [as_of_date] * 2 + ["-", "-", "-"]
})
st.subheader("Overview")
st.table(summary)

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("SPY vs VIX (Normalized to 100)")
    spy_norm = spy["Close"] / spy["Close"].iloc[0] * 100
    vix_norm = vix["Close"] / vix["Close"].iloc[0] * 100
    fig, ax = plt.subplots()
    ax.plot(spy_norm, label="SPY")
    ax.plot(vix_norm, label="VIX")
    ax.set_ylabel("Index (Start = 100)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Daily Return Correlation")
    ret = pd.concat([spy["Returns"], vix["Returns"]], axis=1).dropna()
    ret.columns = ["SPY", "VIX"]
    fig = px.scatter(ret, x="SPY", y="VIX", opacity=0.5)
    st.plotly_chart(fig)

# VIX spike heatmaps
st.subheader("📅 VIX Spike Calendar")
hm2 = vix[vix.Spike == "2SD"].groupby(["Month", "Day"]).size().unstack(fill_value=0)
hm3 = vix[vix.Spike == "3SD"].groupby(["Month", "Day"]).size().unstack(fill_value=0)

st.markdown("**2σ Spikes**")
st.plotly_chart(px.imshow(hm2, labels={"color": "Count"}, title="2σ VIX Spikes"))

st.markdown("**3σ Spikes**")
st.plotly_chart(px.imshow(hm3, labels={"color": "Count"}, title="3σ VIX Spikes"))
