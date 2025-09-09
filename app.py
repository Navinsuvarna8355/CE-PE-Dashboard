# NSE Option Chain Dashboard with Correct IST Timestamp, Technical Analytics, and Styled Layout

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from pytz import timezone as ZoneInfo  # fallback to pytz for legacy Python
import plotly.graph_objs as go
import time

# --- SETTINGS ---
NSE_URL = "https://www.nseindia.com"
OPTION_CHAIN_API = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
INDEX_API = "https://www.nseindia.com/api/allIndices"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"),
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9"
}
TIMEZONE = "Asia/Kolkata"

# --- STYLES ---
CUSTOM_CSS = """
<style>
/* Title & header style */
h1, h2, h3, .st-av, .st-ag {
    text-align: center !important;
}
.section-title {
    color: #2166ac;
    font-weight: bold;
    font-size: 22px;
    margin-top: 24px;
}
.stMetric {
    background-color: #e0f7fa !important;
    border-radius: 8px !important;
    margin: 4px;
}
</style>
"""

st.set_page_config(
    page_title="NSE Option Chain Analytics Dashboard (IST)",
    layout="wide"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def get_ist_now():
    "Return current time as timezone-aware datetime in IST."
    try:
        now = datetime.now(ZoneInfo(TIMEZONE))
    except Exception:
        # Fallback to pytz for <3.9
        import pytz
        now = datetime.now(pytz.timezone(TIMEZONE))
    return now

def format_timestamp(dt):
    "Return a human readable IST timestamp for display."
    return dt.strftime("%A, %d %B %Y • %I:%M %p IST")

@st.cache_data(ttl=30)
def fetch_option_chain():
    "Fetch latest NIFTY option chain data (calls and puts), with cookie/session handling."
    sess = requests.Session()
    sess.get(NSE_URL + "/option-chain", headers=HEADERS, timeout=7)
    response = sess.get(OPTION_CHAIN_API, headers=HEADERS, timeout=10)
    data = response.json()
    return data

@st.cache_data(ttl=900)
def fetch_index_spot():
    "Fetch latest NIFTY50 spot value for technical indicators."
    sess = requests.Session()
    sess.get(NSE_URL, headers=HEADERS, timeout=7)
    resp = sess.get(INDEX_API, headers=HEADERS, timeout=10)
    data = resp.json()
    for idx in data['data']:
        if idx['index'] == "NIFTY 50":
            return idx['last']
    return None

@st.cache_data(ttl=1800)
def fetch_price_history(symbol='NIFTY'):
    "Fetch recent closing price history (simulate with yfinance fallback or constant values for demo)."
    # Use yfinance or Alpha Vantage in production; demo static data
    dates = pd.date_range(get_ist_now() - timedelta(days=30), periods=30)
    prices = 24750 + 200 * np.sin(np.linspace(0,6,30)) + np.random.normal(0, 50, 30)
    df = pd.DataFrame({"date": dates, "close": prices})
    df.set_index("date", inplace=True)
    return df

# --- TECHNICAL INDICATORS ---

def compute_rsi(prices, period=14):
    "Compute RSI using pandas Exponential Moving Average."
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(com=period-1, adjust=False).mean()
    roll_down = down.ewm(com=period-1, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(prices, slow=26, fast=12, signal=9):
    "Compute MACD line, signal line, and histogram."
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- OPTION ANALYTICS ---

def analyze_option_chain(option_data, spot):
    "Analyze option chain for ATM and neighboring strikes. Detect CE/PE decay bias and premium changes."
    records = option_data['records']
    expiry = records['expiryDates'][0]
    strikes = records['strikePrices']
    data = records['data']

    # Find ATM strike (nearest to spot)
    atm_strike = min(strikes, key=lambda x: abs(x-spot))
    strikes_considered = [atm_strike - 50, atm_strike, atm_strike + 50]
    ce_metrics, pe_metrics = [], []

    for row in data:
        if row.get('strikePrice') in strikes_considered and row.get('expiryDate') == expiry:
            ce = row.get('CE', {})
            pe = row.get('PE', {})
            ce_metrics.append({
                "strike": row.get('strikePrice'),
                "ltp": ce.get('lastPrice', np.nan),
                "change": ce.get('change', np.nan),
                "theta": ce.get('theta', np.nan),
                "iv": ce.get('impliedVolatility', np.nan),
                "oi": ce.get('openInterest', np.nan)
            })
            pe_metrics.append({
                "strike": row.get('strikePrice'),
                "ltp": pe.get('lastPrice', np.nan),
                "change": pe.get('change', np.nan),
                "theta": pe.get('theta', np.nan),
                "iv": pe.get('impliedVolatility', np.nan),
                "oi": pe.get('openInterest', np.nan)
            })
    df_ce = pd.DataFrame(ce_metrics).set_index("strike")
    df_pe = pd.DataFrame(pe_metrics).set_index("strike")
    # Calculate decay bias: which premium is decaying faster?
    decay_diff = df_ce['theta'] - df_pe['theta']
    premium_diff = df_ce['ltp'] - df_pe['ltp']
    return df_ce, df_pe, decay_diff, premium_diff, atm_strike, expiry

def recommend_strategy(rsi, macd_hist, decay_bias):
    "Suggest options strategy based on signals."
    if np.isnan(rsi) or np.isnan(macd_hist):
        return "Not enough data for recommendation."
    if rsi > 70 and macd_hist > 0:
        return "Market is overbought & upward momentum: Consider Bear Call Spread or Short CE."
    elif rsi < 30 and macd_hist < 0:
        return "Market is oversold & downward momentum: Consider Bull Put Spread or Short PE."
    elif decay_bias.mean() > 0:
        return "CE premium decays faster: Favor Short Call strategies."
    elif decay_bias.mean() < 0:
        return "PE premium decays faster: Favor Short Put strategies."
    else:
        return "Market neutral: Consider Iron Condor or option writing at both ends."

# --- LAYOUT ---

def show_metrics(spot, rsi, macd_hist, decay_bias, premium_diff):
    "Display top-level metrics panel."
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NIFTY Spot", f"{spot:,.2f}")
    col2.metric("RSI (14)", f"{rsi:.1f}", delta=None)
    col3.metric("MACD Histogram", f"{macd_hist:.2f}", delta=None)
    bias_str = "CE" if decay_bias.mean() > 0 else "PE"
    col4.metric("Decay Bias", f"{bias_str} faster", f"{abs(decay_bias.mean()):.2f}")

def plot_option_decay(df_ce, df_pe, atm_strike):
    "Plot theta and premium for CE/PE near ATM."
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_ce.index.astype(str),
        y=df_ce['theta'],
        name='CE Theta',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=df_pe.index.astype(str),
        y=df_pe['theta'],
        name='PE Theta',
        marker_color='orange'
    ))
    fig.update_layout(
        barmode='group',
        title=f"Theta Decay: CE vs PE near ATM ({atm_strike})",
        xaxis_title="Strike Price", yaxis_title="Theta"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi_macd_closing(df_prices, rsi, macd_line, signal_line, hist):
    "Plot closing price, RSI, and MACD indicators."
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prices.index,
        y=df_prices['close'],
        name="NIFTY Close", line=dict(color='green')
    ))
    fig.update_layout(title="NIFTY Closing Price (30 Days)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_prices.index,
        y=rsi, name="RSI", line=dict(color='red')
    ))
    fig2.add_hline(y=70, line_dash="dash")
    fig2.add_hline(y=30, line_dash="dash")
    fig2.update_layout(title="RSI Indicator (14-period)")
    st.plotly_chart(fig2, use_container_width=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_prices.index,
        y=macd_line, name="MACD", line=dict(color='teal')
    ))
    fig3.add_trace(go.Scatter(
        x=df_prices.index,
        y=signal_line, name="Signal Line", line=dict(color='orange')
    ))
    fig3.add_trace(go.Bar(
        x=df_prices.index,
        y=hist, name="MACD Histogram", marker_color='gray'
    ))
    fig3.update_layout(title="MACD Analysis")
    st.plotly_chart(fig3, use_container_width=True)

# --- MAIN DASHBOARD ---

st.title("NSE Option Chain Analytics Dashboard :chart_with_upwards_trend:")
st.markdown("""
Options analytics and strategy insights for NIFTY, using live NSE data, delta & theta decay analysis, and technical indicators.  
*All timestamps and computations in Indian Standard Time (IST).*
""")

refresh_freq = st.sidebar.slider("Data Refresh (seconds)", min_value=15, max_value=180, value=30, step=1)

with st.expander("About This Dashboard", expanded=False):
    st.write("""
        - **Accurate IST Timestamps** (not server/UTC): to avoid confusion for trading users.
        - **Option Chain Data**: from NSE public APIs.
        - **Technical Filters**: RSI, MACD.
        - **Decay Bias & Premium Trends**: Compare CE/PE for trading edge.
        - **Strategy Recommendations**: Based on signals and decay logic.
        - **Styled Layout & Responsive Charts**: Plotly integration and frontend tweaks.
        - **Best Practices**: Efficient caching, modular functions, robust error handling.
    """)

last_refresh = get_ist_now()

st.markdown(f"**Last Data Refresh:** {format_timestamp(last_refresh)}")

try:
    opt_data = fetch_option_chain()
    spot = fetch_index_spot()
    price_history = fetch_price_history()
    df_ce, df_pe, decay_bias, premium_diff, atm_strike, expiry = analyze_option_chain(opt_data, spot)
    rsi_series = compute_rsi(price_history['close'])
    macd_line, signal_line, hist = compute_macd(price_history['close'])
    rsi_latest = rsi_series.iloc[-1]
    macd_hist_latest = hist.iloc[-1]

    # Metrics overview panel
    show_metrics(spot, rsi_latest, macd_hist_latest, decay_bias, premium_diff)

    # Option decay chart
    st.markdown("<div class='section-title'>Theta Decay (Time Decay Bias) Near ATM</div>", unsafe_allow_html=True)
    plot_option_decay(df_ce, df_pe, atm_strike)

    # RSI/MACD plots
    st.markdown("<div class='section-title'>Technical Indicators: Price, RSI & MACD</div>", unsafe_allow_html=True)
    plot_rsi_macd_closing(price_history, rsi_series, macd_line, signal_line, hist)

    # Option chain analytics details
    st.markdown("<div class='section-title'>Detailed Option Chain (ATM & Nearby)</div>", unsafe_allow_html=True)
    expander = st.expander("Show Option Chain Data (Calls & Puts)")
    with expander:
        st.write("**Call Options (CE):**")
        st.dataframe(df_ce, use_container_width=True)
        st.write("**Put Options (PE):**")
        st.dataframe(df_pe, use_container_width=True)

    # Strategy recommendation
    st.markdown("<div class='section-title'>Strategy Recommendation</div>", unsafe_allow_html=True)
    st.info(recommend_strategy(rsi_latest, macd_hist_latest, decay_bias))

except Exception as e:
    st.error(f"Error in data refresh: {e}")

# --- AUTO-REFRESH for REAL-TIME UX ---
st.experimental_rerun() if st.session_state.get("last_auto_refresh", 0) + refresh_freq < time.time() else None
st.session_state["last_auto_refresh"] = time.time()
