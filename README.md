# Bybit Spot Signal Radar v2

## What's new
- Discovers active Bybit Spot instruments using the V5 instruments endpoint.
- Scans all active USDT-quoted spot pairs, all active spot pairs, or a custom list.
- Timeframe choices: 5m, 15m, 1h, 4h, 1d.
- Copyable signal text using Streamlit's code-block copy control.
- Separate demo / live market-data labels.
- Explicitly does **not** place authenticated orders yet.

## Run
Python 3.10+:
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important
- The all-market scan has a configurable safety cap. Raise it to scan more pairs, but expect several minutes and possible API rate limits.
- If the app reports data warnings, the scan is incomplete. Do not treat missing pairs as "no signal".
- The strategy is an unvalidated research prototype. A score is not a probability of winning.
- Backtesting and execution are not included in this version; do not rely on it for live trades.
- Demo trading and live trading on Bybit require different authenticated setup and endpoint handling. This UI currently does not authenticate and does not submit orders.
- Never paste API keys into the dashboard, source code, chat, or a public hosting service.
- Before adding order execution, implement secrets management, permission checks, demo/live environment verification, quantity rounding using instrument metadata, order status reconciliation, balance checks, idempotency, position/order monitoring, and hard max-order/max-loss limits.
- No bot can guarantee signals, profits, TP hits, or SL avoidance.

## Official API docs
- Spot instruments: https://bybit-exchange.github.io/docs/v5/market/instrument
- Kline data: https://bybit-exchange.github.io/docs/v5/market/kline
- Demo trading: https://bybit-exchange.github.io/docs/v5/demo
- Place order API (future implementation only): https://bybit-exchange.github.io/docs/v5/order/create-order
