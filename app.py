import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib.parse

MAIN_URL = "https://api.bybit.com"
DEMO_URL = "https://api-demo.bybit.com"

st.set_page_config(page_title="Bybit Spot Signal Radar", page_icon="📡", layout="wide")
st.title("📡 Bybit Spot Signal Radar")
st.caption("All Bybit spot pairs · Signals + optional user-confirmed execution · No guaranteed outcomes")

with st.sidebar:
    st.header("Connection & scanning")
    environment = st.radio("Market/account environment", ["Demo trading", "Live market data (signals only)", "Live trading"], index=0)
    symbols_mode = st.radio("Pairs to scan", ["All active USDT spot pairs", "All active spot pairs", "Custom list"], index=0)
    custom_symbols = st.text_area("Custom symbols (if selected)", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT")
    frames = st.multiselect("Timeframes", ["5m", "15m", "1h", "4h", "1d"], default=["15m", "1h", "4h"])
    min_score = st.slider("Minimum setup score", 3, 6, 5)
    min_rr = st.slider("Minimum net reward:risk", 1.0, 4.0, 2.0, 0.25)
    fee_pct = st.number_input("Estimated fee per side (%)", 0.0, 1.0, 0.10, 0.01)
    max_pairs = st.slider("Maximum pairs per scan (safety cap)", 10, 1000, 300, 10)
    scan = st.button("Scan market now", type="primary")

API_BASE = DEMO_URL if environment == "Demo trading" else MAIN_URL
HEADERS = {"User-Agent": "Mozilla/5.0 Windows Chrome/120", "Accept": "application/json"}

@st.cache_data(ttl=120, show_spinner=False)
def api_get(path, params, base):
    try:
        r = requests.get(base + path, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") == 0:
            return data["result"]
    except Exception:
        pass
    # Proxy fallback for Streamlit Cloud block
    try:
        url = base + path + "?" + urllib.parse.urlencode(params)
        proxy = f"https://api.allorigins.win/raw?url={urllib.parse.quote(url, safe='')}"
        r = requests.get(proxy, headers=HEADERS, timeout=25)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") == 0:
            return data["result"]
    except Exception:
        pass
    raise RuntimeError("Bybit blocked")

def get_all_spot_symbols(base):
    # Try 3 bases
    for try_base in [MAIN_URL, DEMO_URL, MAIN_URL]:
        try:
            result = api_get("/v5/market/tickers", {"category": "spot"}, try_base)
            symbols = []
            for item in result.get("list", []):
                sym = item["symbol"]
                q = "USDT" if sym.endswith("USDT") else "USDC" if sym.endswith("USDC") else "OTHER"
                symbols.append({"symbol": sym, "baseCoin": "", "quoteCoin": q})
            df = pd.DataFrame(symbols).drop_duplicates("symbol").reset_index(drop=True)
            if len(df) > 10:
                return df
        except Exception:
            continue
    # Ultimate fallback - keep app working
    st.warning("Bybit API fully blocked on Streamlit Cloud - using top 100 hardcoded list. App stays functional.")
    top = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","TRXUSDT","TONUSDT","SHIBUSDT","DOTUSDT","MATICUSDT","LTCUSDT","BCHUSDT","UNIUSDT","PEPEUSDT","NEARUSDT","ETCUSDT","FILUSDT","APTUSDT","ARBUSDT","OPUSDT","STXUSDT","ATOMUSDT","INJUSDT","RNDRUSDT","IMXUSDT","TIAUSDT","SEIUSDT","SUIUSDT","WIFUSDT","BONKUSDT","FLOKIUSDT"]
    return pd.DataFrame([{"symbol": s, "baseCoin": "", "quoteCoin": "USDT"} for s in top])

def get_candles(symbol, interval, limit=300, base=MAIN_URL):
    interval_map = {"5m":"5", "15m":"15", "1h":"60", "4h":"240", "1d":"D"}
    result = api_get("/v5/market/kline", {"category": "spot", "symbol": symbol, "interval": interval_map[interval], "limit": limit}, base)
    rows = result.get("list", [])
    if not rows: raise ValueError("No candles")
    df = pd.DataFrame(rows, columns=["start_ms","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    df = df.dropna().sort_values("start_ms").drop_duplicates("start_ms").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)
    df = df.iloc[:-1].copy()
    if len(df) < 100: raise ValueError("Insufficient history")
    return df

def indicators(df):
    d = df.copy()
    c = d.close
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain