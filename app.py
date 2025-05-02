import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import morningstar_data as md
from morningstar_data.direct import InvestmentIdentifier

st.title("Live SPY and VIX Analysis")

# — 1) Credentials & Endpoint (US by default) —
md.ApiConfig.username = st.secrets.morningstar.username
md.ApiConfig.password = st.secrets.morningstar.password
# If you’re on EU/APAC, also set md.ApiConfig.base_url accordingly

# — 2) Cached fetch of 10-year Close series —
@st.cache_data(ttl=3600)
def fetch_data():
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")

    # PX_LAST is Morningstar’s ‘closing price’ data point ID
    dp = [{"datapointId": "PX_LAST", "isTsdp": True}]

    inv = [
        InvestmentIdentifier(ticker="SPY"),
        InvestmentIdentifier(ticker="^VIX")
    ]

    df = md.direct.get_investment_data(
        investments=inv,
        data_points=dp,
        time_series_format=md.direct.data_type.TimeSeriesFormat.LONG
    )
    # LONG format → columns: Id, Date, PX_LAST
    df = df.pivot(index="Date", columns="Id", values="PX_LAST")
    df.sort_index(inplace=True)
    df.columns = ["SPY", "VIX"]

    # Compute returns & attach Close
    spy = pd.DataFrame({"Close": df["SPY"]})
    vix = pd.DataFrame({"Close": df["VIX"]})
    spy["Returns"] = spy["Close"].pct_change()
    vix["Returns"] = vix["Close"].pct_change()
    return spy, vix

spy_data, vix_data = fetch_data()

# — 3) Latest Close (last available row) —
latest_spy = spy_data.tail(1)
latest_vix = vix_data.tail(1)

current_spy_price = float(latest_spy["Close"])
current_spy_date  = latest_spy.index[0].strftime("%Y-%m-%d")
current_vix_price = float(latest_vix["Close"])
current_vix_date  = latest_vix.index[0].strftime("%Y-%m-%d")

# — 4) Your existing analysis & charts —
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

st.subheader("Current SPY & VIX Prices…")
# …and then rebuild your price_levels table, line charts, scatter, normalized plots, heatmaps, exactly as before…
