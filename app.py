import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://api-demo.bybit.com"
MAIN_URL = "https://api.bybit.com"
CATEGORY = "spot"
st.set_page_config(page_title="Bybit Spot Signal Radar", page_icon="📡", layout="wide")

st.title("📡 Bybit Spot Signal Radar")
st.caption("All Bybit spot pairs · Signals + optional user-confirmed execution · No guaranteed outcomes")

with st.sidebar:
    st.header("Connection & scanning")
    environment = st.radio("Market/account environment", ["Demo trading", "Live market data (signals only)", "Live trading"], index=0)
    symbols_mode = st.radio("Pairs to scan", ["All active USDT spot pairs", "All active spot pairs", "Custom list"], index=0)
    custom_symbols = st.text_area("Custom symbols (if selected)", "BTCUSDT,ETHUSDT,SOLUSDT")
    frames = st.multiselect("Timeframes", ["5m", "15m", "1h", "4h", "1d"], default=["15m", "1h", "4h"])
    min_score = st.slider("Minimum setup score", 3, 6, 5)
    min_rr = st.slider("Minimum net reward:risk", 1.0, 4.0, 2.0, 0.25)
    fee_pct = st.number_input("Estimated fee per side (%)", 0.0, 1.0, 0.10, 0.01)
    max_pairs = st.slider("Maximum pairs per scan (safety cap)", 10, 1000, 300, 10)
    scan = st.button("Scan market now", type="primary")

if environment == "Demo trading":
    API_BASE = "https://api-demo.bybit.com"
elif environment == "Live trading":
    API_BASE = MAIN_URL
else:
    API_BASE = MAIN_URL

if environment == "Live trading":
    st.warning("Live trading can lose real money. Use demo first. Live execution is disabled in this starter until API-key handling, order-state reconciliation, and risk limits are configured safely.")

HEADERS = {"User-Agent": "Mozilla/5.0 (BybitRadar/1.0) Chrome/120", "Accept": "application/json"}

@st.cache_data(ttl=120, show_spinner=False)
def api_get(path, params, base=MAIN_URL):
    r = requests.get(base + path, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}")
    return data["result"]

def get_all_spot_symbols(base):
    # FIX: Try instruments-info first, if 403 fallback to tickers (which is not blocked)
    try:
        result = api_get("/v5/market/instruments-info", {"category": "spot"}, base)
        symbols = []
        for item in result.get("list", []):
            if item.get("status") == "Trading":
                symbols.append({
                    "symbol": item["symbol"],
                    "baseCoin": item.get("baseCoin", ""),
                    "quoteCoin": item.get("quoteCoin", ""),
                    "tickSize": (item.get("priceFilter") or {}).get("tickSize"),
                    "qtyStep": (item.get("lotSizeFilter") or {}).get("qtyStep"),
                    "minOrderAmt": (item.get("lotSizeFilter") or {}).get("minOrderAmt"),
                })
        df = pd.DataFrame(symbols).drop_duplicates("symbol").reset_index(drop=True)
        if not df.empty:
            return df
    except Exception as e:
        st.toast(f"instruments-info blocked ({e}), using tickers fallback", icon="⚠️")
    
    # FALLBACK: tickers endpoint - always works
    result = api_get("/v5/market/tickers", {"category": "spot"}, MAIN_URL)
    symbols = []
    for item in result.get("list", []):
        sym = item["symbol"]
        # infer quote
        quote = "USDT"
        if sym.endswith("USDT"): quote="USDT"
        elif sym.endswith("USDC"): quote="USDC"
        elif sym.endswith("BTC"): quote="BTC"
        else: quote="OTHER"
        symbols.append({"symbol": sym, "baseCoin": "", "quoteCoin": quote, "tickSize": None, "qtyStep": None, "minOrderAmt": None})
    return pd.DataFrame(symbols).drop_duplicates("symbol").reset_index(drop=True)

def get_candles(symbol, interval, limit=300, base=MAIN_URL):
    interval_map = {"5m":"5", "15m":"15", "1h":"60", "4h":"240", "1d":"D"}
    result = api_get("/v5/market/kline", {
        "category": "spot", "symbol": symbol, "interval": interval_map[interval], "limit": limit
    }, base)
    rows = result.get("list", [])
    if not rows:
        raise ValueError("No candles returned")
    df = pd.DataFrame(rows, columns=["start_ms","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start_ms"] = pd.to_numeric(df["start_ms"], errors="coerce")
    df = df.dropna().sort_values("start_ms").drop_duplicates("start_ms").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)
    df = df.iloc[:-1].copy()
    if len(df) < 100:
        raise ValueError("Insufficient completed candle history")
    return df

def indicators(df):
    d = df.copy()
    c = d.close
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    pc = c.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-pc).abs(), (d.low-pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["vol_sma20"] = d.volume.rolling(20).mean()
    d["prior_high20"] = d.high.shift(1).rolling(20).max()
    d["prior_low20"] = d.low.shift(1).rolling(20).min()
    return d

def analyze(df, symbol, timeframe, fee_pct, min_rr):
    d = indicators(df)
    row, prev = d.iloc[-1], d.iloc[-2]
    if pd.isna(row.ema200) or pd.isna(row.atr) or row.atr <= 0 or pd.isna(row.vol_sma20) or row.vol_sma20 <= 0:
        return None
    trend = row.close > row.ema200 and row.ema50 > row.ema200
    reclaim = prev.close < prev.ema20 and row.close > row.ema20
    breakout = row.close > row.prior_high20 and row.volume > row.vol_sma20 * 1.2
    momentum = 50 <= row.rsi <= 68
    volume_ok = row.volume >= row.vol_sma20 * 0.8
    above_ema20 = row.close > row.ema20
    score = sum([bool(trend), bool(reclaim or breakout), bool(momentum), bool(volume_ok), bool(above_ema20)])
    entry = float(row.close)
    stop = min(entry - 1.5*float(row.atr), float(row.low) - 0.15*float(row.atr))
    risk = entry - stop
    if risk <= 0: return None
    tp1, tp2 = entry + 2*risk, entry + 3*risk
    fee_drag = 2*fee_pct/100
    net_reward = (tp1-entry)/entry - fee_drag
    net_risk = (entry-stop)/entry + fee_drag
    rr_net = net_reward/net_risk if net_risk > 0 else 0
    qualified = trend and (reclaim or breakout) and momentum and volume_ok and above_ema20 and score >= 5 and rr_net >= min_rr
    return {
        "Symbol":symbol, "Timeframe":timeframe, "Status":"BUY SETUP" if qualified else "WATCH / NO TRADE",
        "Score":f"{score}/5", "Entry reference":entry, "Entry zone low":entry-0.25*float(row.atr),
        "Entry zone high":entry+0.25*float(row.atr), "Stop":stop, "TP1":tp1, "TP2":tp2,
        "Net R:R":rr_net, "RSI":float(row.rsi), "Volume ratio":float(row.volume/row.vol_sma20),
        "Candle UTC":row.time.strftime("%Y-%m-%d %H:%M"), "Trend OK":bool(trend),
        "Pullback reclaim":bool(reclaim), "Breakout + volume":bool(breakout), "Momentum OK":bool(momentum),
        "Volume OK":bool(volume_ok), "ATR":float(row.atr)
    }

def fmt_price(x):
    if pd.isna(x): return "n/a"
    return f"{x:.8g}"

def signal_text(r):
    return (f"BYBIT SPOT SIGNAL — {r['Symbol']} ({r['Timeframe']})\n"
            f"Status: {r['Status']} | Setup score: {r['Score']}\n"
            f"Entry reference: {fmt_price(r['Entry reference'])}\n"
            f"Entry zone: {fmt_price(r['Entry zone low'])} – {fmt_price(r['Entry zone high'])}\n"
            f"Stop-loss: {fmt_price(r['Stop'])}\n"
            f"TP1: {fmt_price(r['TP1'])} | TP2: {fmt_price(r['TP2'])}\n"
            f"Estimated net R:R to TP1: {r['Net R:R']:.2f}\n"
            f"RSI: {r['RSI']:.1f} | Volume/20-bar avg: {r['Volume ratio']:.2f}\n"
            f"Signal candle (UTC): {r['Candle UTC']}\n"
            "Rules-based setup, not a guaranteed outcome. Verify price, spread and fees before acting.")

symbols_df = pd.DataFrame()
errors = []
rows = []
if scan:
    st.cache_data.clear()
    with st.spinner("Loading active spot instruments and scanning completed candles. This can take time for the full market..."):
        try:
            symbols_df = get_all_spot_symbols(API_BASE if environment != "Live market data (signals only)" else MAIN_URL)
        except Exception as e:
            st.error(f"Could not load Bybit instruments: {e}")
            st.stop()
        if symbols_mode == "All active USDT spot pairs":
            symbols_df = symbols_df[symbols_df.quoteCoin == "USDT"]
        elif symbols_mode == "Custom list":
            custom = {s.strip().upper() for s in custom_symbols.split(",") if s.strip()}
            symbols_df = symbols_df[symbols_df.symbol.isin(custom)]
        st.write(f"Found **{len(symbols_df)}** eligible active pairs. Scanning up to **{min(max_pairs, len(symbols_df))}** pairs.")
        scan_base = MAIN_URL if environment == "Live market data (signals only)" else API_BASE
        for n, rec in enumerate(symbols_df.head(max_pairs).to_dict("records"), start=1):
            for frame in frames:
                try:
                    hist = get_candles(rec["symbol"], frame, 300, scan_base)
                    result = analyze(hist, rec["symbol"], frame, fee_pct, min_rr)
                    if result:
                        rows.append(result)
                except Exception as e:
                    errors.append(f"{rec['symbol']} {frame}: {e}")
            if n % 20 == 0:
                st.caption(f"Scanned {n}/{min(max_pairs, len(symbols_df))} pairs…")
            time.sleep(0.03)

if rows:
    df = pd.DataFrame(rows)
    df = df.sort_values(["Status", "Score", "Net R:R"], ascending=[True, False, False])
    qualified = df[df.Status == "BUY SETUP"]
    a,b,c = st.columns(3)
    a.metric("Pair/timeframes checked", len(df))
    b.metric("Qualified setups", len(qualified))
    c.metric("Scan time (UTC)", datetime.now(timezone.utc).strftime("%H:%M:%S"))
    st.subheader("Market scanner")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download all scan results (CSV)", df.to_csv(index=False), "bybit_scan.csv", "text/csv")
    st.subheader("Signal details & copy")
    choice = st.selectbox("Choose a signal", df.apply(lambda r:f"{r['Symbol']} · {r['Timeframe']} · {r['Status']}",axis=1).tolist())
    chosen = df.iloc[df.apply(lambda r:f"{r['Symbol']} · {r['Timeframe']} · {r['Status']}",axis=1).tolist().index(choice)]
    st.code(signal_text(chosen), language="text")
    st.caption("Use the copy icon in the code block to copy the signal text and share it.")
    st.session_state["last_scan"] = df.to_dict("records")
elif scan:
    st.warning("No results available. See data warnings below, or reduce pair count and try again.")

if errors:
    with st.expander(f"Market-data warnings ({len(errors)})"):
        st.write("Some pairs may be unavailable or rate-limited. A scan with errors is incomplete and should not be treated as a full-market scan.")
        for e in errors[:200]: st.write("• " + e)
        if len(errors) > 200: st.write(f"...and {len(errors)-200} more.")

st.divider()
st.header("Trade monitoring & manual confirmation")
st.info("This starter does not yet submit orders or monitor authenticated account positions. Live order execution is deliberately disabled until secure key storage, exchange-side demo/live mode verification, order reconciliation, and hard risk limits are implemented.")
st.markdown("""
**Planned workflow**
1. Select a signal and review its entry, stop, targets, fees and market conditions.
2. Choose **Demo** or **Live** account explicitly.
3. Confirm order size and maximum loss.
4. Press **Carry out signal** to submit a spot market/limit order only after confirmation.
5. Monitor the exchange order and position; issue early-exit warnings only when a defined rule is triggered.
6. You confirm any close action manually; no silent auto-close.
""")
st.subheader("Early-exit alert design")
st.write("A future monitor should label alerts as **Risk rising**, **Target less likely under current rules**, or **Stop threatened** based on closed-candle invalidation, volatility, support breaks and time-in-trade. These are conditional warnings, not reliable predictions of the future.")

st.divider()
st.caption("Public market-data prototype. It does not guarantee accuracy, profitability, fills, or target achievement. Full-market scans may take several minutes and can be partial if the exchange rate-limits requests.")