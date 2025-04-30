import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import yfinance as yf
from pandas_datareader import data as pdr

# ---
# Configuration
# ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HIST_WINDOW = "10y"
INTERVAL = "1d"

# ---
# Helpers
# ---
@st.cache_data
def load_historical(ticker: str) -> pd.DataFrame:
    """
    Load historical data from parquet if exists, else fetch and save for future runs.
    """
    file_path = DATA_DIR / f"{ticker}_{HIST_WINDOW}.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
    else:
        df = yf.download(ticker, period=HIST_WINDOW, interval=INTERVAL)
        df.to_parquet(file_path)
    df["Returns"] = df["Close"].pct_change()
    return df

@st.cache_data(ttl=300)
def fetch_latest_eod(tickers: list) -> (dict, pd.Timestamp):
    """
    Fetch most recent EOD close for a list of tickers via Stooq.
    """
    df = pdr.DataReader(tickers, "stooq")
    latest = df.iloc[-1]
    date = df.index[-1]
    return latest.to_dict(), date

# ---
# Load data
# ---
spy = load_historical("SPY")
vix = load_historical("^VIX")

(prices, as_of) = fetch_latest_eod(["SPY", "VIX"])
current_spy_price = float(prices["SPY"])
current_vix_price = float(prices["VIX"])
as_of_date = as_of.strftime("%Y-%m-%d")

# ---
# Calculate metrics
# ---
mu, sigma = vix["Close"].agg(["mean", "std"])
thresh_2sd = mu + 2 * sigma
thresh_3sd = mu + 3 * sigma

vix_percentile = vix["Close"].rank(pct=True).iloc[-1] * 100

vix = vix.assign(
    Spike=lambda df: np.where(df["Close"] >= thresh_3sd, "3SD",
                     np.where(df["Close"] >= thresh_2sd, "2SD", "No Spike"))
)
vix = vix.assign(Month=vix.index.month, Day=vix.index.day)

# ---
# Streamlit App Layout
# ---
st.set_page_config(layout="wide")
st.title("📊 SPY & VIX Dashboard")

# Summary metrics table
summary = pd.DataFrame(
    {
        "Metric": ["Current SPY", "Current VIX", "VIX 2σ", "VIX 3σ", "VIX Percentile"],
        "Value": [
            current_spy_price,
            current_vix_price,
            float(thresh_2sd),
            float(thresh_3sd),
            f"{vix_percentile:.1f}th"
        ],
        "Date": [as_of_date, as_of_date, "-", "-", "-"]
    }
)
st.subheader("Overview")
st.table(summary)

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("SPY vs VIX (100% normalized)")
    spy_norm = spy["Close"] / spy["Close"].iloc[0] * 100
    vix_norm = vix["Close"] / vix["Close"].iloc[0] * 100
    fig, ax = plt.subplots()
    ax.plot(spy_norm, label="SPY")
    ax.plot(vix_norm, label="VIX")
    ax.set_ylabel("Index (Start = 100)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("SPY vs VIX Returns")
    ret = pd.concat([spy["Returns"], vix["Returns"]], axis=1).dropna()
    ret.columns = ["SPY", "VIX"]
    fig = px.scatter(ret, x="SPY", y="VIX", opacity=0.5,
                     title="Daily Return Correlation")
    st.plotly_chart(fig)

# --- Heatmaps ---
st.subheader("📅 VIX Spike Calendar")
hm2 = vix[vix.Spike == "2SD"].groupby(["Month", "Day"]).size().unstack(fill_value=0)
hm3 = vix[vix.Spike == "3SD"].groupby(["Month", "Day"]).size().unstack(fill_value=0)

st.markdown("**2σ Spikes**")
st.plotly_chart(px.imshow(hm2, labels={"color": "Count"}, title="2σ VIX Spikes"))

st.markdown("**3σ Spikes**")
st.plotly_chart(px.imshow(hm3, labels={"color": "Count"}, title="3σ VIX Spikes"))
