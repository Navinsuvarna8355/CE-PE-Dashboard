# app.py

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# For calculating RSI and MACD
import ta

# Setup NSE endpoints and headers
NSE_OC_BASE_URL = "https://www.nseindia.com"
NSE_OC_API = "https://www.nseindia.com/api/option-chain-indices?symbol={}"
NSE_INDEX_API = "https://www.nseindia.com/api/allIndices"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate, br"
}

# Helper function to setup session and handle cookies
def setup_nse_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    # Get cookies by visiting NSE home
    session.get(NSE_OC_BASE_URL, timeout=5)
    return session

# Real-time Fetch of Option Chain Data
def fetch_option_chain(symbol="NIFTY"):
    session = setup_nse_session()
    api_url = NSE_OC_API.format(symbol)
    retries = 0
    while retries < 3:
        response = session.get(api_url, timeout=7)
        if response.status_code == 200:
            data = json.loads(response.text)
            return data
        else:
            retries += 1
            session = setup_nse_session()
    raise Exception("Failed to fetch option chain after 3 attempts.")

# Helper: Get Underlying Price
def fetch_underlying_price(symbol="NIFTY"):
    session = setup_nse_session()
    response = session.get(NSE_INDEX_API, timeout=7)
    data = json.loads(response.text)
    for idx in data['data']:
        if idx['index'] == f"{symbol} 50":
            return float(idx['last'])
        elif symbol == "BANKNIFTY" and idx['index'] == "NIFTY BANK":
            return float(idx['last'])
    return None

# Option Greeks Calculation (Theta)
def calculate_theta(row, underlying_price, expiry_date, typ='CE'):
    """
    Approximate Black-Scholes theta for option. For CE/PE, change sign accordingly.
    Assumptions:
    - European-style
    - ATM as best estimate for IV if missing
    """
    from scipy.stats import norm
    S = underlying_price
    K = row['strikePrice']
    T = row['daysToExpiry'] / 365  # days to year
    r = 0.06  # risk-free, tweak as needed
    d = 0.0   # dividend yield, can fetch if needed
    vol = row['impliedVolatility'] / 100 if row['impliedVolatility'] > 0.1 else 0.20  # as decimal
    if T <= 0:
        return 0
    d1 = (np.log(S / K) + (r - d + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if typ == 'CE':
        theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return theta

# Utility: Days to expiry calculator
def get_days_to_expiry(expiry_str):
    expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
    today = datetime.now()
    delta = (expiry_date - today).days
    return max(0, delta)

# Data Preparation and Analysis
def process_option_chain(option_chain_data, underlying_price):
    """
    Returns a DataFrame with symbol, strike, type, theta, lastPrice, premium change, OI, etc.
    """
    records = []
    expiry_dates = option_chain_data['records']['expiryDates']
    # We take the nearest expiry for real-time bias
    nearest_expiry = expiry_dates[0]

    for d in option_chain_data['records']['data']:
        strike = d['strikePrice']
        # Note: Not every row has both CE and PE (ITM/OTM edges)
        if "CE" in d and d["CE"].get("expiryDate", None) == nearest_expiry:
            ce = d['CE']
            ce_row = {
                "type": "CE",
                "strikePrice": strike,
                "lastPrice": ce.get("lastPrice", 0),
                "change": ce.get("change", 0),
                "impliedVolatility": ce.get("impliedVolatility", 18),
                "openInterest": ce.get("openInterest", 0),
                "changeinOpenInterest": ce.get("changeinOpenInterest", 0),
                "expiryDate": ce.get("expiryDate"),
                "daysToExpiry": get_days_to_expiry(ce.get("expiryDate")),
            }
            ce_row["theta"] = calculate_theta(ce_row, underlying_price, ce_row["expiryDate"], typ='CE')
            records.append(ce_row)
        if "PE" in d and d["PE"].get("expiryDate", None) == nearest_expiry:
            pe = d['PE']
            pe_row = {
                "type": "PE",
                "strikePrice": strike,
                "lastPrice": pe.get("lastPrice", 0),
                "change": pe.get("change", 0),
                "impliedVolatility": pe.get("impliedVolatility", 18),
                "openInterest": pe.get("openInterest", 0),
                "changeinOpenInterest": pe.get("changeinOpenInterest", 0),
                "expiryDate": pe.get("expiryDate"),
                "daysToExpiry": get_days_to_expiry(pe.get("expiryDate")),
            }
            pe_row["theta"] = calculate_theta(pe_row, underlying_price, pe_row["expiryDate"], typ='PE')
            records.append(pe_row)
    df = pd.DataFrame(records)
    return df

# Detect Theta Decay Bias and Suggestions
def detect_theta_bias(df):
    """
    Returns which side experiences steeper theta decay and summarizes bias.
    """
    ce_theta_sum = df[df['type'] == 'CE']["theta"].sum()
    pe_theta_sum = df[df['type'] == 'PE']["theta"].sum()

    bias = None
    if abs(ce_theta_sum) > abs(pe_theta_sum) + 0.5:
        bias = "CE (calls) are decaying faster - Favor short calls or call credit spreads."
    elif abs(pe_theta_sum) > abs(ce_theta_sum) + 0.5:
        bias = "PE (puts) are decaying faster - Favor short puts or put credit spreads."
    else:
        bias = "Balanced decay across CE and PE - Neutral or iron condor strategies may be optimal."
    return {
        'ce_theta_sum': ce_theta_sum,
        'pe_theta_sum': pe_theta_sum,
        'bias': bias
    }

# Premium Change Detection
def premium_change_direction(df):
    ce_prem_chg = df[df['type'] == 'CE'].set_index('strikePrice')['change'].sum()
    pe_prem_chg = df[df['type'] == 'PE'].set_index('strikePrice')['change'].sum()
    if ce_prem_chg < pe_prem_chg - 0.1:
        return "Faster call premium decay detected!"
    elif pe_prem_chg < ce_prem_chg - 0.1:
        return "Faster put premium decay detected!"
    else:
        return "No strong premium change bias."

# Apply RSI and MACD Filters for Trade Signals
def apply_rsi_macd_signals(price_series):
    """
    Returns signal dict based on 14-period RSI and default MACD.
    """
    signals = {}
    # RSI
    rsi = ta.momentum.RSIIndicator(price_series, window=14)
    signals['rsi'] = rsi.rsi()
    # MACD
    macd = ta.trend.MACD(price_series)
    signals['macd'] = macd.macd()
    signals['macd_signal'] = macd.macd_signal()
    signals['macd_diff'] = macd.macd_diff()

    # Entry/Exit logic
    last_rsi = signals['rsi'].iloc[-1]
    last_macd = signals['macd'].iloc[-1]
    last_macd_signal = signals['macd_signal'].iloc[-1]

    signal = "Hold"
    if last_rsi < 30 and last_macd > last_macd_signal:
        signal = "Strong Buy (RSI oversold & MACD bullish cross)"
    elif last_rsi > 70 and last_macd < last_macd_signal:
        signal = "Strong Sell (RSI overbought & MACD bearish cross)"
    elif last_macd > last_macd_signal:
        signal = "MACD bullish cross (consider buying volatility or long strategy)"
    elif last_macd < last_macd_signal:
        signal = "MACD bearish cross (consider selling premium or short strategy)"
    signals['signal'] = signal
    return signals

# Alert System for Sudden Decay Shifts
def detect_sudden_decay_shifts(df, previous_theta_dict):
    alerts = []
    ce_theta_now = df[df['type'] == 'CE']["theta"].sum()
    pe_theta_now = df[df['type'] == 'PE']["theta"].sum()
    threshold = 0.2  # Tune for sensitivity

    if previous_theta_dict:
        ce_jump = abs(ce_theta_now - previous_theta_dict['ce_theta_sum'])
        pe_jump = abs(pe_theta_now - previous_theta_dict['pe_theta_sum'])
        if ce_jump > threshold or pe_jump > threshold:
            alerts.append("ALERT: Sudden jump in theta decay detected for CE or PE!")
    return alerts

# Dash App Layout and Callbacks

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "NSE Option Chain Theta Dashboard"

# Store for session data
app.layout = dbc.Container([
    html.H1("NSE Option Chain Real-Time Theta Decay Dashboard"),
    dbc.Row([
        dbc.Col([
            html.Label("Symbol (NIFTY/BANKNIFTY):"),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[
                    {'label': 'NIFTY', 'value': 'NIFTY'},
                    {'label': 'BANKNIFTY', 'value': 'BANKNIFTY'}
                ],
                value='NIFTY'
            )
        ], width=2),
        dbc.Col([
            html.Button('Manual Fetch', id='manual-fetch-btn', n_clicks=0, className="btn btn-primary"),
            html.Span(id='last-update-time', style={"marginLeft": "10px"})
        ], width=4),
        dbc.Col([
            html.Div("Auto-refresh every 30 seconds."),
            dcc.Interval(id='auto-refresh', interval=30 * 1000, n_intervals=0)
        ], width=2),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='alerts-div', style={"color": "red", "marginTop": "12px"}),
            html.Div(id='strategy-reco-div', style={"color": "blue", "fontWeight": "bold"})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(children=[
                dcc.Graph(id='theta-chart')
            ], color="#119DFF", type="default")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Option Chain Data (Nearest Expiry)"),
            dcc.Loading(children=[
                html.Div(id='option-chain-table')
            ])
        ])
    ]),
    # Hidden store for previous theta values
    dcc.Store(id='prev-theta-sum', data={}),
    # Signal output
    dbc.Row([
        dbc.Col([
            html.H5("RSI/MACD Filtered Trade Signal:"),
            html.Div(id='rsi-macd-signal')
        ])
    ])
], fluid=True)


# Callback for main logic
@app.callback(
    [
        Output('theta-chart', 'figure'),
        Output('option-chain-table', 'children'),
        Output('strategy-reco-div', 'children'),
        Output('alerts-div', 'children'),
        Output('prev-theta-sum', 'data'),
        Output('last-update-time', 'children'),
        Output('rsi-macd-signal', 'children'),
    ],
    [
        Input('auto-refresh', 'n_intervals'),
        Input('manual-fetch-btn', 'n_clicks')
    ],
    State('symbol-dropdown', 'value'),
    State('prev-theta-sum', 'data'),
    prevent_initial_call=True
)
def update_dashboard(auto_interval, manual_fetch, symbol, prev_theta_sum):
    # ----- 1. Fetch option chain and index price -----
    oc_data = fetch_option_chain(symbol)
    underlying_price = fetch_underlying_price(symbol)
    df = process_option_chain(oc_data, underlying_price)
    # ----- 2. Detect theta bias & premium decay -----
    theta_bias_info = detect_theta_bias(df)
    premium_decay_str = premium_change_direction(df)
    strategy_reco = f'''
        <b>Theta Decay Bias:</b> {theta_bias_info["bias"]}
        <br/><b>Premium Change Bias:</b> {premium_decay_str}
        <br/>
        <b>Strategy Tips:</b>
        <ul>
            <li>If calls decay faster, use short call or call credit spreads.</li>
            <li>If puts decay faster, short put or put credit spreads. For neutral bias, try iron condor/strangle.</li>
        </ul>
    '''
    # ----- 3. Theta Plot -----
    fig = go.Figure()
    for typ in ['CE', 'PE']:
        sub = df[df['type'] == typ]
        fig.add_trace(go.Bar(
            x=sub['strikePrice'],
            y=sub['theta'],
            name=typ
        ))
    fig.update_layout(
        title='Theta Decay by Strike (Nearest Expiry)',
        xaxis_title='Strike Price',
        yaxis_title='Theta (per day)',
        barmode='overlay',
        legend=dict(x=0.9, y=1),
        height=420
    )
    # ----- 4. Option Chain Table -----
    tab = df.sort_values(['type', 'strikePrice'])
    table = dbc.Table.from_dataframe(
        tab[['type', 'strikePrice', 'lastPrice', 'change', 'impliedVolatility', 'openInterest', 'theta']],
        striped=True, bordered=True, hover=True
    )
    # ----- 5. Alerts for sudden decay shifts -----
    alerts = detect_sudden_decay_shifts(df, prev_theta_sum)
    # ----- 6. RSI/MACD Signal -----
    # For simplicity, use the synthetic price series: sum of ATM CE and PE last prices as a proxy
    atm_idx = np.abs(df['strikePrice'] - underlying_price).argmin()
    synthetic_series = df[(df['strikePrice'] == df.iloc[atm_idx]['strikePrice'])]\
        .sort_values('type')['lastPrice'].cumsum()
    # Pad to 30 points for indicators, real implementation should use time-series
    price_series = pd.Series(np.repeat(synthetic_series.values, 15))
    signals = apply_rsi_macd_signals(price_series)
    rsi_macd_signal = f"Signal: <b>{signals['signal']}</b> | RSI: {signals['rsi'].iloc[-1]:.2f} | MACD: {signals['macd'].iloc[-1]:.2f}"
    # ----- 7. Package Outputs -----
    return (
        fig, 
        table, 
        strategy_reco, 
        " | ".join(alerts) if alerts else "", 
        {"ce_theta_sum": theta_bias_info['ce_theta_sum'], "pe_theta_sum": theta_bias_info['pe_theta_sum']},
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        rsi_macd_signal
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
