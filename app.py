import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Decay Bias Dashboard", layout="wide")
st.title("⚡ Nifty & BankNifty Decay Bias Dashboard")

# -------------------------------
# Fetch Option Chain Data
# -------------------------------
@st.cache_data(ttl=300)
def fetch_option_chain(symbol="NIFTY"):
    try:
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
        return data["records"]["data"], data["records"]["expiryDates"], data["records"]["underlyingValue"]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return [], [], 0

# -------------------------------
# Process Data
# -------------------------------
def process_data(chain_data):
    rows = []
    for item in chain_data:
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        strike = item.get("strikePrice", ce.get("strikePrice", pe.get("strikePrice", 0)))
        ce_theta = ce.get("theta", 0)
        pe_theta = pe.get("theta", 0)
        ce_change = ce.get("change", 0)
        pe_change = pe.get("change", 0)
        decay_side = "CE" if ce_theta < pe_theta else "PE"
        rows.append({
            "Strike Price": strike,
            "CE Theta": ce_theta,
            "PE Theta": pe_theta,
            "CE Change": ce_change,
            "PE Change": pe_change,
            "Decay Side": decay_side
        })
    return pd.DataFrame(rows)

# -------------------------------
# UI Controls
# -------------------------------
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
refresh = st.sidebar.button("🔄 Refresh Now")

# -------------------------------
# Main Execution
# -------------------------------
chain_data, expiry_list, spot_price = fetch_option_chain(symbol)
if not chain_data:
    st.stop()

df = process_data(chain_data)

# Timestamp in IST
ist = pytz.timezone("Asia/Kolkata")
timestamp = datetime.now(ist).strftime("%A, %d %B %Y • %I:%M %p")

# -------------------------------
# Display Header Info
# -------------------------------
active_bias = df["Decay Side"].value_counts().idxmax()

st.markdown(f"""
**Symbol:** `{symbol}`  
**Spot Price:** `{spot_price}`  
**Expiry:** `{expiry_list[0]}`  
**Decay Bias:** `{active_bias} Decay Active`  
🕒 **Last Updated:** {timestamp}
""")

# -------------------------------
# Display Analysis Table
# -------------------------------
st.subheader("📊 Analysis")
st.dataframe(df, use_container_width=True)

# -------------------------------
# Dynamic Strategy Panel
# -------------------------------
st.subheader("📌 Strategy Recommendations")

if active_bias == "PE":
    st.markdown("""
    #### 🔴 Bearish Bias (Downside)
    - Call options are decaying faster than puts.
    - Consider strategies:
        - 📌 Sell Call Options (Short Call)
        - 📌 Put Options Buying (Long Put)
        - 📌 Bear Put Spread
    """)
elif active_bias == "CE":
    st.markdown("""
    #### 🟢 Bullish Bias (Upside)
    - Put options are decaying faster than calls.
    - Consider strategies:
        - 📌 Sell Put Options (Short Put)
        - 📌 Call Options Buying (Long Call)
        - 📌 Bull Call Spread
    """)
else:
    st.markdown("""
    #### 🟡 Neutral Bias (Range-bound)
    - Low decay on both CE and PE.
    - Consider strategies:
        - 📌 Iron Condor
        - 📌 Straddle
        - 📌 Butterfly Spread
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built by Navinn • Decay Bias Strategy • Powered by NSE Option Chain")
