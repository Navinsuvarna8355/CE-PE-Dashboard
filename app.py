import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="CE/PE Decay Bias Dashboard", layout="wide")
st.title("📈 CE/PE Decay Bias Strategy – Real-Time Dashboard")

# -------------------------------
# NSE Option Chain Fetcher
# -------------------------------
@st.cache_data(ttl=300)
def fetch_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)
    data = response.json()
    return data["records"]["data"]

# -------------------------------
# Strategy Logic
# -------------------------------
def detect_decay(ce_theta, pe_theta):
    if ce_theta < 0 and pe_theta < 0:
        return "CE" if abs(ce_theta) > abs(pe_theta) else "PE"
    elif ce_theta < 0:
        return "CE"
    elif pe_theta < 0:
        return "PE"
    else:
        return "None"

def generate_alerts(row):
    if abs(row["CE Theta"]) > 25 or abs(row["PE Theta"]) > 25:
        return "⚠️ Sudden Decay Shift"
    elif row["RSI"] > 70:
        return "🔼 Overbought"
    elif row["RSI"] < 30:
        return "🔽 Oversold"
    else:
        return ""

# -------------------------------
# Data Processing
# -------------------------------
def process_chain_data(chain_data):
    rows = []
    for item in chain_data:
        if "CE" in item and "PE" in item:
            ce = item["CE"]
            pe = item["PE"]
            strike = ce["strikePrice"]
            ce_theta = ce.get("theta", 0)
            pe_theta = pe.get("theta", 0)
            ce_ltp = ce.get("lastPrice", 0)
            pe_ltp = pe.get("lastPrice", 0)
            rows.append({
                "Strike": strike,
                "CE LTP": ce_ltp,
                "PE LTP": pe_ltp,
                "CE Theta": ce_theta,
                "PE Theta": pe_theta
            })
    df = pd.DataFrame(rows)
    df["Bias"] = df.apply(lambda row: detect_decay(row["CE Theta"], row["PE Theta"]), axis=1)
    df["Strategy"] = df["Bias"].map({
        "CE": "Sell CE / Buy PE",
        "PE": "Sell PE / Buy CE",
        "None": "Iron Condor"
    })
    df["RSI"] = RSIIndicator(close=df["CE LTP"]).rsi()
    macd = MACD(close=df["CE LTP"])
    df["MACD"] = macd.macd_diff()
    df["Alert"] = df.apply(generate_alerts, axis=1)
    return df

# -------------------------------
# UI Controls
# -------------------------------
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
refresh = st.sidebar.button("🔄 Refresh Now")

# -------------------------------
# Main Execution
# -------------------------------
try:
    chain_data = fetch_option_chain(symbol)
    df = process_chain_data(chain_data)
    st.subheader(f"📊 Live Option Chain – {symbol}")
    st.dataframe(df, use_container_width=True)

    # Theta Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["CE Theta"], mode="lines+markers", name="CE Theta", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df["Strike"], y=df["PE Theta"], mode="lines+markers", name="PE Theta", line=dict(color="blue")))
    fig.update_layout(title="Theta Decay by Strike", xaxis_title="Strike Price", yaxis_title="Theta")
    st.plotly_chart(fig, use_container_width=True)

    # Alerts
    st.subheader("⚠️ Strategy Alerts")
    alert_df = df[df["Alert"] != ""]
    st.dataframe(alert_df[["Strike", "Strategy", "Alert"]], use_container_width=True)

except Exception as e:
    st.error("Failed to fetch live data. Please try again later.")
    st.exception(e)

# Footer
st.markdown("---")
st.caption("Built by Navinn • Live Decay Bias Strategy • Powered by NSE Option Chain")
