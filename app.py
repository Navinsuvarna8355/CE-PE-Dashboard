# Real-Time NSE Option Chain Dashboard – CE/PE Theta Decay Bias Detection & Strategy Recommendations
#
# Features:
# - Live NSE option chain fetch with anti-block logic
# - CE/PE theta and premium change tracking (Black-Scholes or mibian/QuantLib as fallback)
# - Plotly chart (live, smooth) for theta decay and premium change
# - Auto-refresh and manual update controls
# - Strategy recommendations (straddle, spreads, etc.) based on detected decay bias
# - Entry/exit timing signals, with RSI/MACD filters
# - Alerts for sudden theta decay shifts using change point detection (ruptures)
# - Modular, clean code with inline documentation

import requests
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import html, dcc, Input, Output, State
import ruptures as rpt
import threading
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
import mibian  # For Black-Scholes Greeks
import sys
import tkinter as tk
from tkinter import messagebox

# -------- CONFIGURATION --------
UNDERLYING = 'NIFTY'  # You may change to 'BANKNIFTY' etc.
EXPIRY = None         # None means auto-pick nearest expiry, else set as 'YYYY-MM-DD'
REFRESH_SECONDS = 60  # Auto-refresh interval
PRELOAD_HISTORY = 10  # How many past windows to store (for decay tracking)
THETA_ALERT_THRESHOLD = 0.2  # Threshold for sudden theta decay alerts
API_HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'accept-language': 'en-US,en;q=0.9',
    'accept-encoding': 'gzip, deflate, br'
}

# -------- GLOBAL STATE --------
data_history = []
last_alert_time = None

# -------- UTILITY FUNCTIONS --------

def fetch_nse_option_chain(symbol):
    """
    Fetch NSE option chain data for an index.
    Returns JSON; handles session and cookies anti-blocking.
    
    """
    url_map = {
        "NIFTY":      "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
        "BANKNIFTY":  "https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY",
        # Add more indices as needed
    }
    session = requests.Session()
    for _ in range(2):
        try:
            # Acquire cookies/session
            session.get("https://www.nseindia.com/option-chain", headers=API_HEADERS, timeout=5)
            resp = session.get(url_map[symbol], headers=API_HEADERS, timeout=7)
            if resp.status_code == 200:
                return resp.json()
            else:
                time.sleep(1)
        except Exception as e:
            print("NSE Option Chain Fetch Error:", e)
            time.sleep(1)
    raise Exception("Failed to fetch option chain from NSE.")

def extract_chain_df(oc_json):
    """
    Parse option chain JSON into DataFrame.
    Only nearest expiry is extracted if not specified.
    
    """
    fut = oc_json['records']['underlyingValue']
    ce_data = []
    pe_data = []
    expiry_dates = [e for e in oc_json['records']['expiryDates']]
    expiry = EXPIRY or expiry_dates[0]
    for row in oc_json['records']['data']:
        strike = row['strikePrice']
        if 'CE' in row and row['expiryDate'] == expiry:
            ce = row['CE']
            ce_data.append({
                'strike': strike,
                'lastPrice': ce.get('lastPrice', 0),
                'openInterest': ce.get('openInterest', 0),
                'changeinOpenInterest': ce.get('changeinOpenInterest', 0),
                'impliedVolatility': ce.get('impliedVolatility', 0),
                'totalTradedVolume': ce.get('totalTradedVolume', 0),
                'bidprice': ce.get('bidprice', 0),
                'askprice': ce.get('askPrice', 0),
                'underlying': fut,
                'spot': fut,
            })
        if 'PE' in row and row['expiryDate'] == expiry:
            pe = row['PE']
            pe_data.append({
                'strike': strike,
                'lastPrice': pe.get('lastPrice', 0),
                'openInterest': pe.get('openInterest', 0),
                'changeinOpenInterest': pe.get('changeinOpenInterest', 0),
                'impliedVolatility': pe.get('impliedVolatility', 0),
                'totalTradedVolume': pe.get('totalTradedVolume', 0),
                'bidprice': pe.get('bidprice', 0),
                'askprice': pe.get('askPrice', 0),
                'underlying': fut,
                'spot': fut,
            })
    ce_df = pd.DataFrame(ce_data)
    pe_df = pd.DataFrame(pe_data)
    ce_df['type'] = 'CE'
    pe_df['type'] = 'PE'
    df = pd.concat([ce_df, pe_df], axis=0).sort_values(['strike', 'type']).reset_index(drop=True)
    return df, expiry, fut

def days_to_expiry(expiry, now=None):
    """Calculate days (fractional) till expiry date."""
    if now is None: now = datetime.now()
    expiry_dt = pd.to_datetime(expiry)
    # Market typically closes at 15:30 IST (10:00 UTC)
    expiry_dt = expiry_dt.replace(hour=15, minute=30)
    dt = (expiry_dt - now)
    return round(dt.total_seconds() / (24 * 3600), 4)

def compute_theta_bs(spot, strike, iv, days, rate=0.065, cp='C'):
    """
    Black-Scholes approximation for theta (per day).
    
    """
    if days <= 0 or iv <= 0:
        return 0.0
    try:
        # Mibian requires volatility as %, days as int
        c = mibian.BS([spot, strike, rate * 100, int(days)], volatility=iv)
        if cp.upper() == 'C':
            th = c.callTheta  # Negative annualized θ
        else:
            th = c.putTheta
        return th / 365  # Normalize theta per day
    except Exception as e:
        return 0

def get_rsi(series, period=14):
    """Compute RSI for given series."""
    rsi = RSIIndicator(series, window=period)
    return rsi.rsi()

def get_macd(series, n_fast=12, n_slow=26, n_signal=9):
    """Return MACD, signal, histogram for series."""
    macd = MACD(series, window_slow=n_slow, window_fast=n_fast, window_sign=n_signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()

def pop_alert(title, message):
    """Show system modal alert (non-blocking)."""
    def show():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    threading.Thread(target=show).start()

# ---------- CE/PE THETA AND PREMIUM CHANGE TRACKING ----------

def process_option_chain(df, fut, expiry, now):
    """
    Adds theta and premium columns; computes CE/PE total theta/premium near ATM.
    Returns result dict with detailed signals.
    
    """
    atm_strike = df.reindex((df['strike']-fut).abs().sort_values().index)['strike'].values[0]
    strk_range = df[(df['strike']>=atm_strike-200) & (df['strike']<=atm_strike+200)]['strike'].unique()
    days = days_to_expiry(expiry, now)
    # Calculate greeks
    thetas, premiums = [], []
    for _, row in df.iterrows():
        th = compute_theta_bs(row['spot'], row['strike'], row['impliedVolatility'], days, cp=row['type'])
        thetas.append(th)
        premiums.append(row['lastPrice'])
    df = df.copy()
    df['theta'] = thetas
    df['premium'] = premiums
    df['relativeStrike'] = df['strike'] - fut
    ce = df[(df['type']=='CE') & (df['strike'].isin(strk_range))].sort_values('strike')
    pe = df[(df['type']=='PE') & (df['strike'].isin(strk_range))].sort_values('strike')
    total_ce_theta = ce['theta'].sum()
    total_pe_theta = pe['theta'].sum()
    total_ce_prem = ce['premium'].sum()
    total_pe_prem = pe['premium'].sum()
    bias = 'CE' if abs(total_ce_theta) > abs(total_pe_theta) else 'PE'
    return {
        'atm_strike': atm_strike,
        'df': df,
        'ce_sum_theta': total_ce_theta,
        'pe_sum_theta': total_pe_theta,
        'ce_sum_premium': total_ce_prem,
        'pe_sum_premium': total_pe_prem,
        'theta_bias': bias,
        'now': now,
        'spot': fut
    }

def append_history(signal):
    """Append latest snapshot to in-memory history (for decay detection)."""
    data_history.append(signal)
    if len(data_history) > PRELOAD_HISTORY:
        data_history.pop(0)

def theta_decay_alert_check(history):
    """Detects sudden decay shifts with ruptures change-point detection."""
    if len(history)<4:
        return False, None
    ce_thetas = [h['ce_sum_theta'] for h in history]
    pe_thetas = [h['pe_sum_theta'] for h in history]
    arr = np.array([ce_thetas, pe_thetas]).T
    model = rpt.Pelt(model="rbf").fit(arr)
    # Penalization tuned for short windows
    bkps = model.predict(pen=THETA_ALERT_THRESHOLD)
    sudden = len(bkps) > 1
    return sudden, bkps if sudden else None

# ------------ STRATEGY RECOMMENDATION LOGIC -------------------

def recommend_strategy(signal):
    """
    Recommend option strategy based on theta bias and decay rate.
    
    """
    ce, pe = abs(signal['ce_sum_theta']), abs(signal['pe_sum_theta'])
    ce_prem, pe_prem = signal['ce_sum_premium'], signal['pe_sum_premium']
    ratio = ce / (pe+1e-8)
    strategy, expl = None, ""
    if abs(ce-pe) < 0.25*max(ce,pe):
        strategy = "Short Straddle"
        expl = "CE and PE theta balanced—rangebound bias. Write both CE and PE ATM options for maximum theta decay."
    elif ratio > 1.2:
        strategy = "Sell CE, Avoid PE"
        expl = "Call options decaying faster—bearish conditions. Safer to write CE, avoid/small size PE writing."
    elif 1/ratio > 1.2:
        strategy = "Sell PE, Avoid CE"
        expl = "Put options decaying faster—bullish bias. Favor PE writing, minimize CE exposure."
    elif (ce+pe)<5:
        strategy = "Avoid Selling Options"
        expl = "Extremely low theta decay. Premium erosion is slow; avoid writing until volatility/premium improves."
    else:
        strategy = "Iron Condor (wide strikes)"
        expl = "Theta decay on both sides, minor skew. Favor OTM iron condor for hedged premium harvesting."
    return strategy, expl

# ------------ ENTRY/EXIT & MOMENTUM FILTERS -------------------

def entry_exit_signals(signal, price_series):
    """
    Use RSI/MACD as additional filter for entries/exits.
    """
    rsi_val = get_rsi(price_series)[-1]
    macd, macd_signal, macd_hist = get_macd(price_series)
    macd_last, macd_sig_last = macd.iloc[-1], macd_signal.iloc[-1]
    entry = exit = None
    rsi_msg = ""
    macd_msg = ""
    # Filter logic: Entry only if RSI in neutral range & MACD not diverging
    if 40 < rsi_val < 65 and abs(macd_last - macd_sig_last) < 0.5:
        entry = True
        rsi_msg = "RSI neutral."
        macd_msg = "MACD confirms trendless regime."
    else:
        entry = False
        if rsi_val >= 70:
            rsi_msg = "RSI overbought—avoid fresh CE writing."
        elif rsi_val <= 30:
            rsi_msg = "RSI oversold—avoid fresh PE writing."
        else:
            rsi_msg = "RSI not in optimal zone."
        if macd_last > macd_sig_last:
            macd_msg = "MACD indicates bullish push."
        elif macd_last < macd_sig_last:
            macd_msg = "MACD signals downside."
    return entry, exit, rsi_msg, macd_msg

# ------------ DASHBOARD INITIALIZATION AND CALLBACKS ------------

# Price Series for RSI/MACD: For demo, maintain rolling spot history.
spot_series = [np.nan] * 30

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(f"NSE Option Chain Real-Time Theta Decay Dashboard ({UNDERLYING})"),
    html.Div(id="status-bar"),
    html.Div([
        html.A(html.Button('Manual Refresh'),href='/'),  # Forces page reload; or use a dedicated callback
        html.Div(id="last-updated"),
        html.Label("Auto Refresh (seconds):"),
        dcc.Input(id='refresh-interval', type='number', min=20, max=300, value=REFRESH_SECONDS, step=10),
    ], style={"display":"flex","gap":"15px"}),
    dcc.Interval(id='interval-component', interval=REFRESH_SECONDS*1000, n_intervals=0),
    dcc.Graph(id='theta-premium-graph'),
    html.H4("Recommendations"),
    html.Div(id='recommendations'),
    html.H4("Entry/Exit Momentum Filter"),
    html.Div(id='momentum-signals'),
    html.H4("Alerts"),
    html.Div(id='alert-bar')
])

@app.callback(
    Output('interval-component', 'interval'),
    Input('refresh-interval', 'value')
)
def update_interval(seconds):
    """Dynamically adjust auto-refresh frequency."""
    if not seconds or seconds < 20:
        return 60000
    return int(seconds*1000)

@app.callback(
    [
        Output('theta-premium-graph', 'figure'),
        Output('status-bar','children'),
        Output('last-updated','children'),
        Output('recommendations', 'children'),
        Output('momentum-signals', 'children'),
        Output('alert-bar','children')
    ],
    [
        Input('interval-component', 'n_intervals')
    ]
)
def update_dashboard(n):
    now = datetime.now()
    try:
        chain_json = fetch_nse_option_chain(UNDERLYING)
        df, expiry, fut = extract_chain_df(chain_json)
        signal = process_option_chain(df, fut, expiry, now)
        spot_series.append(signal['spot'])
        if len(spot_series) > 120:
            spot_series.pop(0)
        append_history(signal)
        # Chart
        ce_df = df[df['type']=="CE"].sort_values('strike')
        pe_df = df[df['type']=="PE"].sort_values('strike')
        # Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ce_df['strike'], y=ce_df['theta'],
                                 mode='lines+markers', name='CE Theta (per day)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=pe_df['strike'], y=pe_df['theta'],
                                 mode='lines+markers', name='PE Theta (per day)', line=dict(color='red')))
        fig.add_trace(go.Bar(x=ce_df['strike'], y=ce_df['premium'], name='CE Premium',
                             marker=dict(color='lightblue'), opacity=0.5, yaxis="y2"))
        fig.add_trace(go.Bar(x=pe_df['strike'], y=pe_df['premium'], name='PE Premium',
                             marker=dict(color='salmon'), opacity=0.5, yaxis="y2"))
        fig.update_layout(
            title="Theta Decay and Premium vs Strike Price",
            xaxis_title="Strike Price",
            yaxis_title="Theta (per day)",
            yaxis2=dict(
                title="Premium",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h")
        )
        # Recommendations
        strategy, expl = recommend_strategy(signal)
        # Entry/Exit
        entry, exit, rsi_msg, macd_msg = entry_exit_signals(signal, pd.Series(spot_series).dropna())
        entry_status = "Entry Permitted" if entry else "Entry Blocked"
        # Alert
        sudden, bkps = theta_decay_alert_check(data_history)
        alert_msg = ""
        if sudden:
            global last_alert_time
            if not last_alert_time or (now-last_alert_time).total_seconds()>300:
                alert_msg = "🔔 Sudden Theta Decay Shift detected! Review hedge and open positions immediately."
                pop_alert("Theta Decay Alert", alert_msg)
                last_alert_time = now
        status = f"Spot={signal['spot']:.2f} | Expiry: {expiry} | ATM: {signal['atm_strike']}"
        lastupd = f"Last update: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        # Recommendations display
        recmd_html = html.Div([
            html.B(strategy),
            html.Div(expl),
            html.Div(f"CE Theta: {signal['ce_sum_theta']:.2f} | PE Theta: {signal['pe_sum_theta']:.2f}"),
            html.Div(f"CE Premium: {signal['ce_sum_premium']:.2f} | PE Premium: {signal['pe_sum_premium']:.2f}"),
            html.Div(f"Decay Bias: {signal['theta_bias']}")
        ])
        # Entry/Exit/Momentum
        msignal_html = html.Div([
            html.Li(f"Entry/Exit Timing: {entry_status}"),
            html.Li(f"RSI: {rsi_msg}"),
            html.Li(f"MACD: {macd_msg}")
        ])
        # Alerts
        alert_bar = html.B(alert_msg, style={"color":"red"}) if alert_msg else ""
        return [fig, status, lastupd, recmd_html, msignal_html, alert_bar]
    except Exception as e:
        # Error state fallback
        return [go.Figure(), f"Error: {e}", "", "", "", ""]

# -------------- RUN APP ---------------

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


