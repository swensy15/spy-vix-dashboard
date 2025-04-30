import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import yfinance as yf
import requests
from io import StringIO
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# --- Configuration ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HIST_WINDOW = "10y"
INTERVAL = "1d"
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "YOUR_NEW_API_KEY")  # Use new key
CACHE_TTL = 3600  # 1 hour cache for real-time data

# --- Helpers ---
@st.cache_data
def load_historical_spy() -> pd.DataFrame:
    """
    Load 10-year SPY history from Parquet if present;
    otherwise fetch via Alpha Vantage or Stooq and save.
    """
    fp = DATA_DIR / f"SPY_{HIST_WINDOW}.parquet"
    try:
        if fp.exists():
            df = pd.read_parquet(fp)
            if not df.empty and df.index[-1] >= datetime.now() - timedelta(days=2):
                return df
    except Exception as e:
        st.warning(f"Failed to load cached SPY data: {e}")

    # Try Alpha Vantage first
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        df, _ = ts.get_daily_adjusted(symbol="SPY", outputsize="full")
        df.columns = ["Open", "High", "Low", "Close", "Adjusted Close", "Volume", "Dividend", "Split"]
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df = df.last("10y")  # Limit to 10 years
        df.to_parquet(fp)
        df["Returns"] = df["Close"].pct_change()
        return df
    except Exception as e:
        st.warning(f"Alpha Vantage failed: {e}")

    # Fallback to Stooq
    try:
        url = f"https://stooq.com/q/d/l/?s=spy.us&i=d"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"]).set_index("Date")
        df.columns = [c.capitalize() for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.last("10y")
        df.to_parquet(fp)
        df["Returns"] = df["Close"].pct_change()
        return df
    except Exception as e:
        st.error(f"Stooq failed: {e}")
        # Last resort: yfinance
        df = yf.download("SPY", period=HIST_WINDOW, interval=INTERVAL, progress=False)
        df.to_parquet(fp)
        df["Returns"] = df["Close"].pct_change()
        return df

@st.cache_data
def load_historical_vix() -> pd.DataFrame:
    """
    Load full VIX history via Stooq CSV (no rate limits).
    """
    try:
        url = "https://stooq.com/q/d/l/?s=vix&i=d"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"]).rename(
            columns=lambda c: c.capitalize()
        ).set_index("Date")
        df["Returns"] = df["Close"].pct_change()
        return df
    except Exception as e:
        st.error(f"Failed to load VIX data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def fetch_latest_spy() -> (float, pd.Timestamp):
    """
    Fetch the most recent EOD SPY close via Stooq CSV.
    Returns (price, as_of_date).
    """
    try:
        url = "https://stooq.com/q/l/?s=spy.us&f=sd2t2ohlc&h&e=csv"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        temp = pd.read_csv(StringIO(resp.text), parse_dates=["Date"])
        last = temp.iloc[-1]
        return float(last["Close"]), last["Date"]
    except Exception as e:
        st.warning(f"Stooq latest SPY failed: {e}")
        # Fallback to Alpha Vantage
        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
            df, _ = ts.get_daily(symbol="SPY", outputsize="compact")
            last = df.iloc[-1]
            return float(last["4. close"]), pd.to_datetime(df.index[-1])
        except Exception as e:
            st.error(f"Alpha Vantage latest SPY failed: {e}")
            return None, None

# --- Load data ---
spy = load_historical_spy()
vix = load_historical_vix()

if spy.empty or vix.empty:
    st.error("Failed to load required data. Please try again later.")
    st.stop()

current_spy_price, as_of_dt = fetch_latest_spy()
if current_spy_price is None:
    st.error("Failed to fetch latest SPY price.")
    st.stop()

current_vix_price = float(vix["Close"].iloc[-
