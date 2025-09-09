# trading_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
from datetime import datetime, timedelta

# OPTIONAL: Use for real-time Indian options data and Greeks
# from nse_greeks_calculator import NSEGreeksCalculator

# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Options Trading Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>h1 {margin-bottom: 0.5em;} .block-container {padding-top:1.2em;} .metric {font-size: 1.1em;} .stMetricValue {font-size: 2em;} </style>",
    unsafe_allow_html=True,
)

# -- Assets and Default Settings
DEFAULT_SYMBOL = "AAPL"
SUPPORTED_SYMBOLS = ["AAPL", "MSFT", "GOOG", "TSLA", "NIFTY", "BANKNIFTY"]

# Timeframe for percent change and charting
DEFAULT_LOOKBACK_MINUTES = 60  # 1 hour

# -- Option Expiry Handling Example (mock)
DEFAULT_EXPIRIES = [
    (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
    for d in range(0, 31, 7)
]

# ------------------------------------------------------------
# UTILITY FUNCTIONS: DATA FETCH, TECHNICAL INDICATORS, GREEKS
# ------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_spot_price(symbol):
    """Fetch the latest spot price from Yahoo Finance or Binance (fallback)"""
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            timeout=3,
        ).json()
        return response["chart"]["result"][0]["meta"]["regularMarketPrice"]
    except Exception:
        # Fallback example using Binance API for non-NSE USDT pairs
        try:
            binance_symbol = symbol.upper() + "USDT"
            data = requests.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}",
                timeout=2,
            ).json()
            return float(data["price"])
        except Exception:
            return np.nan

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def mock_option_chain(symbol, expiry):
    """Mock option chain with random Greek values, prices. Replace with live feed as needed."""
    strikes = np.arange(0.95, 1.05, 0.01) * fetch_spot_price(symbol)
    ce_prices = np.random.uniform(5, 15, len(strikes))
    pe_prices = np.random.uniform(7, 13, len(strikes))
    ce_theta = np.random.uniform(-1.8, -0.7, len(strikes))
    pe_theta = np.random.uniform(-1.2, -0.5, len(strikes))
    ce_iv = np.random.uniform(15, 22, len(strikes))
    pe_iv = np.random.uniform(14, 23, len(strikes))
    df = pd.DataFrame({
        "Strike": strikes.astype(int),
        "CE_Price": ce_prices,
        "PE_Price": pe_prices,
        "CE_Theta": ce_theta,
        "PE_Theta": pe_theta,
        "CE_IV": ce_iv,
        "PE_IV": pe_iv,
    })
    return df

@st.cache_data(ttl=300)
def fetch_historical_prices(symbol, minutes=120):
    """Fetch historical price data for asset, past X minutes (mocked for demonstration)"""
    now = datetime.now()
    times = pd.date_range(now - timedelta(minutes=minutes), now, periods=minutes)
    base = fetch_spot_price(symbol)
    # Synthetic walk:
    changes = np.random.normal(0, base*0.002, len(times))
    prices = base + np.cumsum(changes)
    return pd.DataFrame({"Datetime": times, "Close": prices})

# --------------------------------------------------------------
# STRATEGY AND SIGNAL: BIAS SCORING, FILTER STATUS, FAST COMPARISON
# --------------------------------------------------------------

def infer_bias_and_confidence(rsi, macd, spot_change_pct, ce_theta, pe_theta):
    """Rule-based bias/confidence scoring (user-adjustable)"""
    score = 50  # start neutral

    # RSI signals
    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15

    # MACD signal
    if macd > 0:
        score += 15
    elif macd < 0:
        score -= 15

    # Spot price trend
    if spot_change_pct > 0:
        score += 10
    elif spot_change_pct < 0:
        score -= 10

    # Theta decay comparison
    theta_diff = ce_theta - pe_theta
    if theta_diff < -0.2:
        score -= 10  # Calls decay faster, more bearish
    elif theta_diff > 0.2:
        score += 10  # Puts decay faster, more bullish

    # Clamp
    score = np.clip(score, 0, 100)

    if score >= 65:
        bias = "Bullish"
    elif score <= 35:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return bias, int(score)

# ----------------------------- UI LAYOUTS & APP -----------------------------

# --- Sidebar Controls:

with st.sidebar:
    st.header("Asset and Expiry")
    symbol = st.selectbox("Symbol", SUPPORTED_SYMBOLS, index=0)
    expiry = st.selectbox("Option Expiry", DEFAULT_EXPIRIES)
    lookback = st.slider("Percent Change Lookback (minutes)", 15, 240, DEFAULT_LOOKBACK_MINUTES, step=15)
    rsi_period = st.slider("RSI Period", 7, 21, 14)
    macd_fast = st.number_input("MACD Fast EMA", min_value=5, max_value=30, value=12)
    macd_slow = st.number_input("MACD Slow EMA", min_value=10, max_value=50, value=26)
    macd_signal = st.number_input("MACD Signal EMA", min_value=5, max_value=20, value=9)

# --- MAIN DASHBOARD HEADER: Live Spot, Expiry, Bias

st.title("💹 Options Trading Dashboard")
cols_top = st.columns([2, 2, 2, 1])

with cols_top[0]:
    spot_price = fetch_spot_price(symbol)
    st.metric("Live Spot Price", f"{spot_price:.2f}")
with cols_top[1]:
    st.metric("Option Expiry", expiry)
with cols_top[2]:
    last_fetch = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"<span style='color:gray'>Last update: {last_fetch}</span>", unsafe_allow_html=True)

# --- DATA RETRIEVAL AND CALCULATION (All Core Elements) ---

# Historical prices
hist = fetch_historical_prices(symbol, minutes=lookback+30)
hist = hist.reset_index(drop=True)
hist['RSI'] = calculate_rsi(hist["Close"], period=rsi_period)
hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = calculate_macd(
    hist["Close"],
    fast=macd_fast,
    slow=macd_slow,
    signal=macd_signal,
)
hist.dropna(inplace=True)

# Price change %
spot_change = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100

# Latest RSI, MACD values
rsi_now = hist["RSI"].iloc[-1]
macd_now = hist["MACD_Hist"].iloc[-1]

# Option chain/Greeks (mock, replace with live feed if available)
option_df = mock_option_chain(symbol, expiry)
atm_row = option_df.iloc[(option_df['Strike'] - spot_price).abs().argmin()]
ce_theta, pe_theta = atm_row["CE_Theta"], atm_row["PE_Theta"]

# --- STRATEGY LOGIC: Bias and Confidence

bias, confidence = infer_bias_and_confidence(rsi_now, macd_now, spot_change, ce_theta, pe_theta)
col_bias = cols_top[3]
with col_bias:
    st.metric("Bias", bias, f"{confidence}%", delta_color="normal")
    st.progress(confidence / 100.0, text=f"{bias.upper()} ({confidence}%)")

# --- SECONDARY ROW: CE/PE Theta, Price % Change, RSI, MACD

cols_stats = st.columns(4)

# 1. CE vs PE Theta
with cols_stats[0]:
    st.caption("ATM CE vs PE Theta Decay")
    st.metric("CE Theta", f"{ce_theta:.2f}")
    st.metric("PE Theta", f"{pe_theta:.2f}")
    theta_delta = ce_theta - pe_theta
    st.markdown(f"Δ Theta: **{theta_delta:.2f}**")

# 2. Price % Change
with cols_stats[1]:
    st.caption("Spot Price Change (%)")
    st.metric("Change (%)", f"{spot_change:.2f}%")

# 3. RSI Filter
with cols_stats[2]:
    st.caption(f"RSI (Period: {rsi_period})")
    rsi_label = "Oversold" if rsi_now < 30 else ("Overbought" if rsi_now > 70 else "Neutral")
    rsi_color = "green" if rsi_now < 30 else ("red" if rsi_now > 70 else "gray")
    st.metric("Current RSI", f"{rsi_now:.1f}")
    st.markdown(f"**Status:** :{rsi_color}[{rsi_label}]", unsafe_allow_html=True)

# 4. MACD Filter
with cols_stats[3]:
    st.caption("MACD Histogram")
    macd_label = "Bullish" if macd_now > 0 else "Bearish" if macd_now < 0 else "Neutral"
    macd_color = "green" if macd_now > 0 else ("red" if macd_now < 0 else "gray")
    st.metric("MACD", f"{macd_now:.3f}")
    st.markdown(f"**Signal:** :{macd_color}[{macd_label}]", unsafe_allow_html=True)

# --- CHARTS AND DEEP DIVE ---

st.subheader("Candlestick Chart & Indicators")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist["Datetime"],
        y=hist["Close"],
        mode="lines",
        name="Spot Price",
        line=dict(color="blue"),
    )
)
fig.add_trace(
    go.Scatter(
        x=hist["Datetime"],
        y=hist["RSI"],
        mode="lines",
        name="RSI",
        yaxis="y2",
        line=dict(color="orange", dash="dot"),
    )
)
fig.update_layout(
    yaxis=dict(domain=[0.3, 1]),
    yaxis2=dict(
        title="RSI",
        range=[0, 100],
        overlaying="y",
        side="right",
        domain=[0, 0.29],
        showgrid=False,
    ),
    title="Spot Price & RSI Trend",
    legend=dict(orientation="h"),
    margin=dict(l=0, r=0, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# --- TABLE: Option Chain and Greeks (ATM Highlight) ---

st.subheader("ATM Option Chain Comparison")
highlight = (option_df['Strike'] - spot_price).abs().idxmin()
option_df_disp = option_df.copy()
option_df_disp["ATM"] = ""
option_df_disp.loc[highlight, "ATM"] = "<-- ATM"

st.dataframe(
    option_df_disp.style.apply(
        lambda x: [
            "background-color: #ffc107" if atm else "" for atm in x["ATM"]
        ],
        axis=1,
    ),
    use_container_width=True,
    hide_index=True,
)

# --- EXPLANATIONS AND STRATEGY LOGIC ---

with st.expander("Strategy Details & Bias Score Explanation", expanded=False):
    st.write(
        """
        **Bias confidence score combines the following:**
        - Recent price trend (percent change over the selected period)
        - RSI (Relative Strength Index - oversold/overbought status)
        - MACD Histogram value (momentum direction)
        - Theta decay competition between ATM CE and PE
        ---
        Each indicator contributes a weighted value to an overall percentage, informing the suggested bias (Bullish, Bearish, Neutral).
        """
    )

# --------------------- END OF APP ---------------------

