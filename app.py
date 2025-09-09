# trading_dashboard.py
# Streamlit Options and Spot Trading Dashboard
# Includes: live spot price, expiry, CE/PE theta comparison, price change %, RSI, MACD, bias confidence score
# Mobile-friendly, fast-scan layout

import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import numpy as np
import datetime
import time
import yfinance as yf  # Optional fallback for non-live data

# ========== CONFIGURATION & UTILS ==========

# --- Streamlit Page Config for Wide, Mobile-Responsive Layout ---
st.set_page_config(
    page_title="Trading Dashboard - Spot & Options Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Use custom CSS for more mobile tuning if desired
st.markdown(
    """
    <style>
        .reportview-container .main .block-container{
            padding-top: 0rem;
            padding-bottom: 0rem;
            max-width: 900px; /* For mobile scaling */
        }
        [data-testid="stMetric"] > div {
            font-size: 1.6em;
        }
        /* Sidebar width tweak for mobile usability */
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 280px;
        }
    </style>
    """, unsafe_allow_html=True
)
# ========== API HANDLING ==========

# Polygon.io API key (replace with your own)
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"

# API endpoints
POLYGON_BASE = "https://api.polygon.io"

def get_spot_price(symbol: str):
    """Fetch latest spot price for symbol from Polygon.io."""
    url = f"{POLYGON_BASE}/v2/last/trade/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        last = data['results']['p']  # Last trade price
        ts = data['results']['t'] / 1000
        return last, datetime.datetime.fromtimestamp(ts)
    except Exception as e:
        st.warning(f"Live spot price not available ({str(e)}). Using fallback data.")
        # As fallback, use yfinance
        t = yf.Ticker(symbol)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1]), hist.index[-1].to_pydatetime()
        return None, None

@st.cache_data(ttl=3600)
def get_option_expiries(symbol):
    """Get list of expiries for this stock/ETF."""
    url = f"{POLYGON_BASE}/v3/reference/options/contracts?underlying_ticker={symbol}&limit=50&apiKey={POLYGON_API_KEY}"
    r = requests.get(url)
    contracts = r.json().get("results", [])
    expiries = sorted({c['expiration_date'] for c in contracts})
    return expiries or []

@st.cache_data(ttl=300)
def get_options_chain(symbol, expiry):
    """Download full options chain for this expiry."""
    url = f"{POLYGON_BASE}/v3/snapshot/options/{symbol}?expiration_date={expiry}&apiKey={POLYGON_API_KEY}"
    r = requests.get(url)
    return r.json().get("results", {})

@st.cache_data(ttl=900)
def get_prev_close(symbol):
    """Get previous day close (for change % calculation)."""
    t = yf.Ticker(symbol)
    hist = t.history(period="2d")
    if len(hist) >= 2:
        return float(hist['Close'].iloc[-2])
    return None

@st.cache_data(ttl=300)
def get_historical_data(symbol, period="2mo", interval="1d"):
    t = yf.Ticker(symbol)
    try:
        df = t.history(period=period, interval=interval)
        return df
    except Exception:
        return pd.DataFrame()

# ========== UI: SIDEBAR CONTROLS ==========

with st.sidebar:
    st.header("⏳ Dashboard Filters")
    symbol = st.text_input("Stock/ETF Ticker", value="AAPL", max_chars=8)
    st.caption("e.g. AAPL, MSFT, SPY, NIFTY")

    # Expiry selection updates on symbol change
    if symbol:
        expiries = get_option_expiries(symbol)
        expiry = st.selectbox("Select Option Expiry", options=expiries)
    else:
        expiry = None

    # RSI/MACD filter thresholds (user tunable)
    st.markdown("**Technical Indicator Filters**")
    rsi_high = st.slider("RSI Overbought", min_value=60, max_value=90, value=70)
    rsi_low = st.slider("RSI Oversold", min_value=5, max_value=40, value=30)
    macd_fast = st.number_input("MACD Fast EMA", min_value=6, max_value=20, value=12)
    macd_slow = st.number_input("MACD Slow EMA", min_value=14, max_value=40, value=26)
    macd_signal = st.number_input("MACD Signal", min_value=5, max_value=20, value=9)

    st.caption("Adjust filter thresholds for strategy tuning.")
    refresh = st.button("🔄 Refresh Data")

# ========== MAIN AREA: LAYOUT & DATA DISPLAY ==========

st.title("📈 Fast Trading Dashboard")
st.caption("Spot & Options | Clean, Mobile-First Streamlit Layout")

# 1. Live Spot Price and Price Change %
spot_price, spot_time = get_spot_price(symbol)
prev_close = get_prev_close(symbol)

price_change_pct = None
if spot_price and prev_close:
    price_change_pct = ((spot_price - prev_close) / prev_close) * 100

# Style metrics row: spot price and change
col1, col2 = st.columns([2, 1])
with col1:
    st.metric("Live Spot Price", f"${spot_price:.2f}" if spot_price else "–", 
              help=f"Last updated: {spot_time.strftime('%Y-%m-%d %H:%M:%S') if spot_time else '–'}")
with col2:
    change_color = "normal"
    if price_change_pct is not None:
        change_color = "inverse" if price_change_pct < 0 else "normal"
    st.metric(
        "Price Change %", 
        f"{price_change_pct:+.2f}%", 
        delta_color=change_color,
        help=f"vs prev close ${prev_close:.2f}" if prev_close else "N/A"
    )

st.markdown("---")

# 2. CE/PE Theta Decay and Expiry Row
if expiry and spot_price:
    options_chain = get_options_chain(symbol, expiry)
    calls = [o for o in options_chain.get("calls", [])]
    puts = [o for o in options_chain.get("puts", [])]
    # Find ATM strike (nearest to spot)
    strikes = [c['strike_price'] for c in calls]
    if strikes:
        atm_strike = min(strikes, key=lambda x: abs(x-spot_price))
        ce = next((c for c in calls if c['strike_price'] == atm_strike), calls[0])
        pe = next((p for p in puts if p['strike_price'] == atm_strike), puts[0])
        ce_theta = ce.get('greeks', {}).get('theta')
        pe_theta = pe.get('greeks', {}).get('theta')
    else:
        atm_strike, ce, pe, ce_theta, pe_theta = None, None, None, None, None
else:
    ce_theta = pe_theta = atm_strike = None

colce, colpe, colexp = st.columns(3)
with colce:
    st.metric("Call (CE) Theta Decay", f"{ce_theta:.3f}" if ce_theta else "–", help="ATM call theta decay")
with colpe:
    st.metric("Put (PE) Theta Decay", f"{pe_theta:.3f}" if pe_theta else "–", help="ATM put theta decay")
with colexp:
    st.metric("Expiry", expiry if expiry else "–", help="Current option series expiry")
st.caption(f"ATM Strike: {atm_strike}" if atm_strike else "")

st.markdown("---")

# 3. Technical Filters: RSI, MACD, Bias Score

hist = get_historical_data(symbol, period="2mo", interval="1d")
if not hist.empty:
    hist = hist.dropna()
    # RSI and MACD using pandas-ta
    hist['rsi'] = ta.rsi(hist['Close'], length=14)
    macd_df = ta.macd(hist['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    hist = pd.concat([hist, macd_df], axis=1)
    # Get latest indicator values
    latest_rsi = hist['rsi'].iloc[-1]
    latest_macd = hist['MACD_12_26_9'].iloc[-1]
    latest_signal = hist['MACDs_12_26_9'].iloc[-1]

    # Determine indicator filter status
    rsi_flag = ("⬆️ Bullish", "🟡 Neutral", "⬇️ Bearish")
    if latest_rsi < rsi_low:
        rsi_status = rsi_flag[2]
    elif latest_rsi > rsi_high:
        rsi_status = rsi_flag[0]
    else:
        rsi_status = rsi_flag[1]
    macd_status = "⬆️ Bullish" if latest_macd > latest_signal else "⬇️ Bearish"

    # Bias score: +1 each for bullish RSI or MACD; –1 for bearish; sum
    bias_score = 0
    if rsi_status == "⬆️ Bullish":
        bias_score += 1
    elif rsi_status == "⬇️ Bearish":
        bias_score -= 1
    if macd_status == "⬆️ Bullish":
        bias_score += 1
    else:
        bias_score -= 1
    # Price change contribution
    if price_change_pct is not None and abs(price_change_pct) > 1.0:
        bias_score += np.sign(price_change_pct)
    # Theta skew: if calls decaying faster and bias bullish, +0.5
    if ce_theta is not None and pe_theta is not None:
        if ce_theta < pe_theta:
            bias_score += 0.5
        else:
            bias_score -= 0.5

    # Normalize to 0–100 bull/bear (0 is strong bear, 100 strong bull, 50 is neutral)
    bias_display = int(50 + bias_score * 12)  # Bias score ranges from –3.5 to 3.5
    bias_label = (
        "🟢 Bullish" if bias_display > 65 else
        "🔴 Bearish" if bias_display < 35 else
        "🟡 Neutral"
    )
else:
    rsi_status, latest_rsi, macd_status, bias_display, bias_label = "–", None, "–", 50, "–"

colrsi, colmacd, colscore = st.columns(3)
with colrsi:
    st.metric("RSI (14)", f"{latest_rsi:.1f}" if latest_rsi else "–", delta=rsi_status)
with colmacd:
    st.metric("MACD", f"{latest_macd:.2f}" if hist is not None else "–", delta=macd_status)
with colscore:
    st.metric("Bias Confidence", f"{bias_display}/100", delta=bias_label,
              help="Composite of RSI, MACD, price, theta")

st.markdown("---")

# 4. Interactive Charts and Details (Expander)

with st.expander("📊 View Historical Chart & Indicators"):
    if not hist.empty:
        # Price line + RSI and MACD subplot
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name="Price"))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['rsi'], name="RSI",
            yaxis="y2", line=dict(color='orange')))
        # Add MACD lines to 2nd y-axis
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MACD_12_26_9'], name="MACD",
            yaxis="y3", line=dict(color='blue')))
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MACDs_12_26_9'], name="Signal",
            yaxis="y3", line=dict(color='red')))
        fig.update_layout(
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
            yaxis3=dict(title="MACD", anchor="free", overlaying="y", side="right", position=1.0, showgrid=False),
            height=540,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough historical data for charting.")

# 5. Option Chain Table (partial, for ATM row)

with st.expander("🗒️ At-the-Money Option Details"):
    if ce and pe:
        d = {
            "Strike": [atm_strike],
            "Call LTP": [ce.get('last_price')],
            "Put LTP": [pe.get('last_price')],
            "Call OI": [ce.get('open_interest')],
            "Put OI": [pe.get('open_interest')],
            "Call Theta": [ce_theta],
            "Put Theta": [pe_theta],
        }
        st.table(pd.DataFrame(d))
    else:
        st.info("Options chain details not available for this expiry/strike.")

st.markdown("---")
st.caption("Powered by Streamlit, Polygon.io, yfinance, pandas-ta. Layout and UI optimized for mobile and rapid decisions.")

# Optional: auto-refresh logic (every N seconds if needed)
if refresh:
    st.experimental_rerun()

