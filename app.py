import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

API = "https://api.bybit.com"
st.set_page_config(page_title="Spot Signal Radar", page_icon="📡", layout="centered", initial_sidebar_state="collapsed")
st.markdown(""" <style> .block-container{max-width:900px;padding:1rem 1rem 3rem} h1{font-size:clamp(1.6rem,5vw,2.3rem)!important} div.stButton>button,div.stDownloadButton>button{width:100%;min-height:3rem;border-radius:.8rem;font-weight:700} div[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.2);padding:.7rem;border-radius:.8rem} </style>""", unsafe_allow_html=True)
st.title("📡 Spot Signal Radar")
st.caption("Bybit Spot · Mobile-friendly · Public market data only")
st.info("Rule-based setups are not guaranteed predictions. This app does not place trades or monitor your account.", icon="🛡️")

@st.cache_data(ttl=120, show_spinner=False)
def api_get(path, params):
    r = requests.get(API + path, params=params, timeout=20, headers={"User-Agent":"SpotSignalRadar/1.0"})
    r.raise_for_status()
    j = r.json()
    if j.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {j.get('retCode')}: {j.get('retMsg')}")
    return j["result"]

@st.cache_data(ttl=300, show_spinner=False)
def get_symbols():
    data = api_get("/v5/market/instruments-info", {"category":"spot"})
    rows = [{"symbol":x["symbol"],"quoteCoin":x.get("quoteCoin","")} for x in data.get("list",[]) if x.get("status")=="Trading"]
    if not rows: raise RuntimeError("Bybit returned no active Spot instruments.")
    return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)

def candles(symbol, tf, limit=300):
    intervals={"5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}
    data=api_get("/v5/market/kline",{"category":"spot","symbol":symbol,"interval":intervals[tf],"limit":limit})
    df=pd.DataFrame(data.get("list",[]),columns=["ms","open","high","low","close","volume","turnover"])
    if df.empty: raise ValueError("No candle data")
    for c in ["ms","open","high","low","close","volume","turnover"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna().sort_values("ms").drop_duplicates("ms").reset_index(drop=True)
    df["time"]=pd.to_datetime(df["ms"],unit="ms",utc=True)
    df=df.iloc[:-1].copy() # ignore still-forming candle
    if len(df)<220: raise ValueError(f"Only {len(df)} completed candles; need at least 220")
    return df

def analyse(df,symbol,tf,fee,min_score,min_rr):
    d=df.copy(); c=d.close
    d["ema20"]=c.ewm(span=20,adjust=False).mean(); d["ema50"]=c.ewm(span=50,adjust=False).mean(); d["ema200"]=c.ewm(span=200,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
    pc=c.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/14,adjust=False).mean(); d["vavg"]=d.volume.rolling(20).mean(); d["prior_high"]=d.high.shift(1).rolling(20).max()
    r,p=d.iloc[-1],d.iloc[-2]
    if any(pd.isna(r[k]) for k in ["ema20","ema50","ema200","rsi","atr","vavg","prior_high"]) or r.atr<=0 or r.vavg<=0 or r.close<=0: return None
    trend=bool(r.close>r.ema200 and r.ema50>r.ema200)
    reclaim=bool(p.close<=p.ema20 and r.close>r.ema20)
    breakout=bool(r.close>r.prior_high and r.volume>r.vavg*1.2)
    momentum=bool(50<=r.rsi<=68); volume_ok=bool(r.volume>=r.vavg*.8); above=bool(r.close>r.ema20)
    score=sum([trend,bool(reclaim or breakout),momentum,volume_ok,above])
    entry=float(r.close); stop=min(entry-1.5*float(r.atr),float(r.low)-.15*float(r.atr)); risk=entry-stop
    if risk<=0:return None
    tp1=entry+2*risk; tp2=entry+3*risk; fee_drag=2*fee/100
    net_reward=(tp1-entry)/entry-fee_drag; net_risk=(entry-stop)/entry+fee_drag
    rr=net_reward/net_risk if net_risk>0 else 0
    qualified=bool(trend and (reclaim or breakout) and momentum and volume_ok and above and score>=min_score and rr>=min_rr)
    return {"Symbol":symbol,"Timeframe":tf,"Status":"QUALIFIED SETUP" if qualified else "WATCH — NOT QUALIFIED","Setup score":f"{score}/5","Entry reference":entry,"Stop-loss reference":stop,"TP1 reference":tp1,"TP2 reference":tp2,"Estimated net R:R to TP1":round(rr,2),"RSI":round(float(r.rsi),1),"Volume / 20-bar avg":round(float(r.volume/r.vavg),2),"Signal candle (UTC)":r.time.strftime("%Y-%m-%d %H:%M")}

def price(x): return f"{float(x):.8g}"
def signal_text(r):
    return (f"BYBIT SPOT SIGNAL — {r['Symbol']} ({r['Timeframe']})\nStatus: {r['Status']} | Score: {r['Setup score']}\nEntry reference: {price(r['Entry reference'])}\nStop-loss reference: {price(r['Stop-loss reference'])}\nTP1 reference: {price(r['TP1 reference'])}\nTP2 reference: {price(r['TP2 reference'])}\nEstimated net R:R: {r['Estimated net R:R to TP1']:.2f}\nRSI: {r['RSI']} | Volume ratio: {r['Volume / 20-bar avg']}\nCandle UTC: {r['Signal candle (UTC)']}\nNot financial advice; check spread, fees and price before acting.")

with st.container(border=True):
    st.subheader("Scan settings")
    with st.form("scan"):
        mode=st.selectbox("Pairs to scan",["Active USDT Spot pairs","All active Spot pairs","Custom symbols"])
        custom=st.text_input("Custom symbols (comma-separated)","BTCUSDT,ETHUSDT,SOLUSDT")
        frames=st.multiselect("Timeframes",["5m","15m","1h","4h","1d"],default=["15m","1h","4h"])
        a,b=st.columns(2)
        with a: min_score=st.select_slider("Minimum setup score",[3,4,5],value=5)
        with b: min_rr=st.select_slider("Minimum estimated net R:R",[1.0,1.25,1.5,2.0,2.5,3.0],value=2.0)
        fee=st.number_input("Estimated fee per side (%)",0.0,1.0,.10,.01)
        cap=st.select_slider("Maximum pairs per scan",[10,25,50,100,150,300,500],value=50)
        go=st.form_submit_button("🔎 Scan Market",type="primary",use_container_width=True)

if go:
    st.cache_data.clear()
    try:
        with st.spinner("Loading active Bybit Spot markets…"): symbols=get_symbols()
    except Exception as e:
        st.error(f"Could not load Bybit market data: {e}"); st.info("Check your connection and try again. No signal was generated."); st.stop()
    if mode=="Active USDT Spot pairs": symbols=symbols[symbols.quoteCoin=="USDT"]
    elif mode=="Custom symbols":
        wanted={s.strip().upper() for s in custom.split(",") if s.strip()}; symbols=symbols[symbols.symbol.isin(wanted)]
    if not frames: st.warning("Choose at least one timeframe."); st.stop()
    if symbols.empty: st.warning("No matching active pairs found. Check your symbols."); st.stop()
    pairs=symbols.head(cap).to_dict("records"); total=len(pairs)*len(frames); done=0; results=[]; errors=[]
    bar=st.progress(0,text="Preparing scan…"); note=st.empty()
    for rec in pairs:
        for tf in frames:
            note.caption(f"Checking {rec['symbol']} · {tf} ({done+1}/{total})")
            try:
                row=analyse(candles(rec["symbol"],tf),rec["symbol"],tf,fee,min_score,min_rr)
                if row: results.append(row)
            except Exception as e: errors.append(f"{rec['symbol']} {tf}: {e}")
            done+=1; bar.progress(done/total,text=f"Scan progress: {done}/{total}"); time.sleep(.08)
    note.empty(); bar.empty()
    st.session_state["results"]=results; st.session_state["errors"]=errors
    st.session_state["summary"]=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC} · {len(pairs)} pairs × {len(frames)} timeframes"

if "summary" in st.session_state:
    st.subheader("Latest scan"); st.caption(st.session_state["summary"])
    results=st.session_state.get("results",[]); errors=st.session_state.get("errors",[])
    df=pd.DataFrame(results)
    if not df.empty:
        good=df[df.Status=="QUALIFIED SETUP"]
        a,b=st.columns(2); a.metric("Qualified setups",len(good)); b.metric("Checks with errors",len(errors))
        st.caption("A qualified setup passes these rules; it is not a win probability.")
        cols=["Symbol","Timeframe","Status","Setup score","Entry reference","Stop-loss reference","TP1 reference","TP2 reference","Estimated net R:R to TP1","RSI"]
        if not good.empty:
            st.subheader("Qualified setups"); st.dataframe(good[cols].reset_index(drop=True),use_container_width=True,hide_index=True)
        else: st.warning("No setups passed all filters. Do not loosen filters just to force a signal.")
        with st.expander("View all analysed results"): st.dataframe(df[cols].reset_index(drop=True),use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download scan results (CSV)",df.to_csv(index=False).encode(), "bybit_spot_scan_results.csv","text/csv",use_container_width=True)
        st.subheader("Signal details & copy")
        labels=[f"{r['Symbol']} · {r['Timeframe']} · {r['Status']}" for r in results]
        chosen=results[labels.index(st.selectbox("Choose a result",labels))]
        st.code(signal_text(chosen),language="text")
        st.caption("Use the copy icon in the signal box to copy and share the signal.")
    else: st.warning("No analysable results. Check warnings below; this is not a BUY signal.")
    if errors:
        with st.expander(f"Market-data warnings ({len(errors)})"):
            st.write("This scan may be incomplete; do not treat it as a full-market scan.")
            for e in errors[:100]: st.write("• "+e)
            if len(errors)>100: st.write(f"…and {len(errors)-100} more")

with st.expander("How to read the signals"):
    st.markdown("- **Setup score /5:** counts five rule conditions; it is not a win percentage.\n- **Entry, stop and targets:** reference levels from the latest completed candle and may become stale.\n- **Net R:R:** approximate fee adjustment only; spread, slippage and actual fills may differ.\n- **Qualified setup:** passes the selected filters but can still lose money.\n- **Warnings:** any data errors mean the scan may be incomplete.")
st.caption("Public market data only. No API keys, trading, account-position monitoring, verified profitability, or guaranteed accuracy. Backtest and paper-trade before risking money.")