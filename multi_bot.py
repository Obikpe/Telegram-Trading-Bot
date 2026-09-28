"""
Multi-asset ranked Telegram alert bot (v3).

Live server : gunicorn multi_bot:app --workers 1 --threads 4
Backtest    : python multi_bot.py backtest 180
Env vars    : TELEGRAM_BOT_TOKEN, PREMIUM_CHANNEL_ID, DB_PATH, ENABLED_ASSETS (comma list, optional)
Long history: put data/<ASSET>_30m.csv (columns: time,open,high,low,close[,volume], UTC) to backtest
              beyond Yahoo's 60-day limit on 30m data (e.g. exported from MetaTrader 5 or Dukascopy).
"""
import os
import sys
import time
import sqlite3
import threading
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from flask import Flask, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("multibot")
app = Flask(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHANNEL = os.getenv("PREMIUM_CHANNEL_ID", "")
DB_PATH = os.getenv("DB_PATH", "multi_signals_v2.db")  # new schema, new file
BINANCE = ["https://api.binance.com", "https://data-api.binance.vision"]

# cost_pct  = assumed ROUND-TRIP cost (spread+fees+slippage) as % of price. Set to your broker's real numbers.
# atr_pct   = allowed range of 30m ATR as % of price (differs hugely per market).
# group     = correlated markets; only one open signal per group at a time.
ASSETS = {
    "BTCUSDT": dict(src="binance", sym="BTCUSDT", dec=2, cost_pct=0.12, atr_pct=(0.25, 3.0),
                    has_vol=True, group="crypto", feed="Binance spot"),
    "ETHUSDT": dict(src="binance", sym="ETHUSDT", dec=2, cost_pct=0.14, atr_pct=(0.30, 3.5),
                    has_vol=True, group="crypto", feed="Binance spot"),
    "EURUSD": dict(src="yahoo", sym="EURUSD=X", dec=5, cost_pct=0.010, atr_pct=(0.03, 0.40),
                   has_vol=False, group="usd_fx", feed="Yahoo EURUSD=X (indicative, may lag)"),
    "GBPUSD": dict(src="yahoo", sym="GBPUSD=X", dec=5, cost_pct=0.012, atr_pct=(0.03, 0.45),
                   has_vol=False, group="usd_fx", feed="Yahoo GBPUSD=X (indicative, may lag)"),
    "XAUUSD": dict(src="yahoo", sym="GC=F", dec=2, cost_pct=0.020, atr_pct=(0.07, 1.2),
                   has_vol=False, group="metals", feed="COMEX gold futures via Yahoo (broker spot differs)"),
    "XAGUSD": dict(src="yahoo", sym="SI=F", dec=3, cost_pct=0.050, atr_pct=(0.12, 2.0),
                   has_vol=False, group="metals", feed="COMEX silver futures via Yahoo (broker spot differs)"),
}
ENABLED = [x.strip() for x in os.getenv("ENABLED_ASSETS", ",".join(ASSETS)).split(",") if x.strip() in ASSETS]

CFG = dict(
    adx_min=22, vol_ratio_min=1.1,
    sl_atr_pad=0.2, risk_min_atr=0.8, risk_max_atr=2.5,
    tp1_r=1.5, tp2_r=3.0, max_hold=96,
    min_score=0,         # ranking score floor; raise only after /stats shows what high scores actually earn
    max_per_cycle=2,     # max alerts per scan, from different correlation groups
    max_chase_r=0.25,    # tell subscribers to skip if price already moved this many R past entry
)
HOUR_MS = 3_600_000

# ------------------------------------------------------------------ data layer
def _closed(df):
    now = int(time.time() * 1000)
    return df[df.close_time < now].reset_index(drop=True)

def fetch_binance(symbol, interval, limit=500, start_ms=None):
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "q", "n", "tb", "tq", "x"]
    rows, cursor = [], start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000 if start_ms else limit}
        if cursor:
            params["startTime"] = cursor
        data = None
        for base in BINANCE:
            try:
                r = requests.get(f"{base}/api/v3/klines", params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    break
            except requests.RequestException as e:
                log.warning("binance %s: %s", base, e)
        if not data:
            break
        rows += data
        if not start_ms or len(data) < 1000:
            break
        cursor = data[-1][6] + 1
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return _closed(df.drop_duplicates("open_time")[["open_time", "open", "high", "low", "close", "volume", "close_time"]])

def fetch_yahoo(ticker, interval, period, minutes):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    except Exception as e:
        log.warning("yahoo %s failed: %s", ticker, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index, utc=True)
    ms = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    out = pd.DataFrame({
        "open_time": np.asarray(ms, dtype="int64"),
        "open": df["Open"].to_numpy(float), "high": df["High"].to_numpy(float),
        "low": df["Low"].to_numpy(float), "close": df["Close"].to_numpy(float),
        "volume": df["Volume"].to_numpy(float) if "Volume" in df else 0.0,
    }).dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["close_time"] = out.open_time + minutes * 60_000 - 1  # Yahoo stamps candle START
    return _closed(out.drop_duplicates("open_time"))

def load_csv(path, days=None):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df["time"], utc=True)
    df["open_time"] = np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1), dtype="int64")
    if "volume" not in df:
        df["volume"] = 0.0
    df["close_time"] = df.open_time + 30 * 60_000 - 1
    df = _closed(df.sort_values("open_time")[["open_time", "open", "high", "low", "close", "volume", "close_time"]])
    if days:
        df = df[df.open_time >= df.open_time.iloc[-1] - days * 86_400_000].reset_index(drop=True)
    return df

def resample(df, minutes):
    """Build higher-timeframe candles (Yahoo has no 4h)."""
    if df.empty:
        return df
    ms = minutes * 60_000
    g = df.assign(b=df.open_time // ms * ms).groupby("b")
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum")).reset_index().rename(columns={"b": "open_time"})
    out["close_time"] = out.open_time + ms - 1
    return _closed(out)

_cache4h = {}
def get_frames(name, days=None):
    """Returns (df4h, df30). `days` is only set by the backtester."""
    a = ASSETS[name]
    csv = os.path.join("data", f"{name}_30m.csv")
    if days and os.path.exists(csv):
        df30 = load_csv(csv, days)
        return resample(df30, 240), df30
    now = int(time.time() * 1000)
    if a["src"] == "binance":
        if days:
            start = now - days * 86_400_000
            return (fetch_binance(a["sym"], "4h", start_ms=start - 60 * 86_400_000),
                    fetch_binance(a["sym"], "30m", start_ms=start))
        return fetch_binance(a["sym"], "4h", limit=500), fetch_binance(a["sym"], "30m", limit=500)
    df30 = fetch_yahoo(a["sym"], "30m", "59d" if days else "20d", 30)
    hit = _cache4h.get(name)
    if not days and hit and time.time() - hit[0] < 1500:
        return hit[1], df30
    h1 = fetch_yahoo(a["sym"], "60m", "700d" if days else "120d", 60)
    df4h = resample(h1, 240)
    if not days:
        _cache4h[name] = (time.time(), df4h)
    return df4h, df30

# ------------------------------------------------------------------ indicators
def rma(s, n): return s.ewm(alpha=1 / n, adjust=False).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / (dn + 1e-12))

def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)

def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi, mdi = 100 * rma(plus, n) / a, 100 * rma(minus, n) / a
    return rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12), n)

def build_frame(df4h, df30, a):
    if df4h.empty or df30.empty:
        return pd.DataFrame()
    h = df4h.copy()
    h["h4_ema50"], h["h4_ema200"], h["h4_adx"] = ema(h.close, 50), ema(h.close, 200), adx(h)
    h["h4_close"] = h.close
    h["h4_high20"], h["h4_low20"] = h.high.rolling(20).max(), h.low.rolling(20).min()
    h["avail"] = h.close_time + 1  # usable only after the 4H candle closes (no look-ahead)
    h = h[["avail", "h4_close", "h4_ema50", "h4_ema200", "h4_adx", "h4_high20", "h4_low20"]]

    f = df30.copy()
    f["ema21"], f["ema50"] = ema(f.close, 21), ema(f.close, 50)
    f["rsi"], f["atr"] = rsi(f.close), atr(f)
    f["atr_pct"] = 100 * f.atr / f.close
    macd = ema(f.close, 12) - ema(f.close, 26)
    f["hist"] = macd - ema(macd, 9)
    f["vol_ratio"] = f.volume / f.volume.rolling(20).mean() if a["has_vol"] else 1.0
    f["swing_low"], f["swing_high"] = f.low.rolling(10).min(), f.high.rolling(10).max()
    f["touch_long"] = (f.low <= f.ema21).astype(int).rolling(8).max() == 1
    f["touch_short"] = (f.high >= f.ema21).astype(int).rolling(8).max() == 1
    f["rsi_min8"], f["rsi_max8"] = f.rsi.rolling(8).min().shift(1), f.rsi.rolling(8).max().shift(1)
    return pd.merge_asof(f, h, left_on="close_time", right_on="avail", direction="backward")

# ------------------------------------------------------------------ strategy + ranking score
def evaluate(name, f, i, a):
    if f.empty or i < 250 or i >= len(f):
        return None
    r, p = f.iloc[i], f.iloc[i - 1]
    need = ["h4_ema200", "h4_adx", "atr", "rsi", "ema50", "rsi_min8", "vol_ratio", "h4_high20", "h4_low20", "hist"]
    if r[need].isna().any():
        return None
    lo, hi = a["atr_pct"]

    # Use bracket notation for safe dictionary/series extraction
    r_atr_pct = float(r["atr_pct"])
    r_h4_adx = float(r["h4_adx"])
    r_h4_close = float(r["h4_close"])
    r_h4_ema200 = float(r["h4_ema200"])
    r_h4_ema50 = float(r["h4_ema50"])
    r_ema21 = float(r["ema21"])
    r_ema50 = float(r["ema50"])
    r_rsi = float(r["rsi"])
    p_rsi = float(p["rsi"])
    r_hist = float(r["hist"])
    p_hist = float(p["hist"])
    r_close = float(r["close"])
    r_open = float(r["open"])
    p_high = float(p["high"])
    p_low = float(p["low"])
    r_vol_ratio = float(r["vol_ratio"]) if a["has_vol"] else 1.0
    r_swing_low = float(r["swing_low"])
    r_swing_high = float(r["swing_high"])
    r_atr = float(r["atr"])
    r_rsi_min8 = float(r["rsi_min8"])
    r_rsi_max8 = float(r["rsi_max8"])
    r_h4_high20 = float(r["h4_high20"])
    r_h4_low20 = float(r["h4_low20"])
    r_close_time = int(r["close_time"])

    if not (lo <= r_atr_pct <= hi) or r_h4_adx < CFG["adx_min"]:
        return None
    if a["has_vol"] and r_vol_ratio < CFG["vol_ratio_min"]:
        return None

    bull = r_h4_close > r_h4_ema200 and r_h4_ema50 > r_h4_ema200
    bear = r_h4_close < r_h4_ema200 and r_h4_ema50 < r_h4_ema200
    d = 0
    if (bull and r_ema21 > r_ema50 and r["touch_long"] and r_rsi_min8 < 42
            and p_rsi < 45 <= r_rsi and r_hist > p_hist and r_close > r_open and r_close > p_high):
        d = 1
    elif (bear and r_ema21 < r_ema50 and r["touch_short"] and r_rsi_max8 > 58
          and p_rsi > 55 >= r_rsi and r_hist < p_hist and r_close < r_open and r_close < p_low):
        d = -1
    if d == 0:
        return None

    entry = r_close
    risk = (entry - (r_swing_low - CFG["sl_atr_pad"] * r_atr)) if d == 1 else ((r_swing_high + CFG["sl_atr_pad"] * r_atr) - entry)
    risk = max(risk, CFG["risk_min_atr"] * r_atr)
    if risk > CFG["risk_max_atr"] * r_atr:
        return None

    # Ranking score (0-100): compares setups across markets. NOT a probability of winning.
    clip = lambda x: min(1.0, max(0.0, x))
    adx_s = clip((r_h4_adx - CFG["adx_min"]) / 18)
    depth = (42 - r_rsi_min8) if d == 1 else (r_rsi_max8 - 58)
    rsi_s = clip(depth / 20)
    body_s = clip(abs(r_close - r_open) / (1.2 * r_atr))
    vol_s = clip((r_vol_ratio - CFG["vol_ratio_min"]) / 1.0) if a["has_vol"] else body_s
    score = 35 * adx_s + 25 * rsi_s + 20 * body_s + 20 * vol_s
    room = d * ((r_h4_high20 if d == 1 else r_h4_low20) - entry)  # distance to nearest 4H barrier
    blocked = 0 < room < CFG["tp1_r"] * risk
    if blocked:
        score -= 15

    return dict(
        symbol=name, d=d, entry=entry, sl=entry - d * risk,
        tp1=entry + d * CFG["tp1_r"] * risk, tp2=entry + d * CFG["tp2_r"] * risk,
        risk=risk, score=round(max(0, min(100, score))), time=r_close_time,
        cost_pct=a["cost_pct"],
        why=(f"4H trend {'up' if d == 1 else 'down'} (ADX {r_h4_adx:.0f}), pullback to 30m EMA21, "
             f"RSI reclaim {r_rsi:.0f}, MACD turning"
             + (f", volume x{r_vol_ratio:.1f}" if a["has_vol"] else "")
             + (", 4H barrier close to TP1" if blocked else "")),
    )

def resolve(s, c):
    """Half off at TP1 (stop to entry), rest at TP2. SL assumed first if both hit in one candle."""
    d, entry, sl, stage = s["d"], s["entry"], s["sl"], 0
    cost = s["cost_pct"] / 100 * entry / s["risk"]
    half1, half2 = 0.5 * CFG["tp1_r"], 0.5 * CFG["tp2_r"]
    for n, (hi, lo, cl, ct) in enumerate(zip(c.high, c.low, c.close, c.close_time), 1):
        hit_sl = lo <= sl if d == 1 else hi >= sl
        hit_tp1 = hi >= s["tp1"] if d == 1 else lo <= s["tp1"]
        hit_tp2 = hi >= s["tp2"] if d == 1 else lo <= s["tp2"]
        if stage == 0:
            if hit_sl:
                return "SL", -1 - cost, int(ct), n
            if hit_tp1:
                stage, sl = 1, entry
        else:
            if hit_sl:
                return "TP1_BE", half1 - cost, int(ct), n
            if hit_tp2:
                return "TP2", half1 + half2 - cost, int(ct), n
        if n >= CFG["max_hold"]:
            move = d * (cl - entry) / s["risk"]
            return "EXPIRED", (move if stage == 0 else half1 + 0.5 * move) - cost, int(ct), n
    return None

# ------------------------------------------------------------------ stats + storage
def summarize(rs):
    if not rs:
        return dict(trades=0)
    a = np.array(rs)
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    cv = np.cumsum(a)
    return dict(trades=len(a), win_rate=round(100 * (a > 0).mean(), 1), avg_r=round(float(a.mean()), 2),
                total_r=round(float(a.sum()), 2), profit_factor=round(float(w / l), 2) if l else None,
                max_drawdown_r=round(float((np.maximum.accumulate(cv) - cv).max()), 2))

def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS signals(
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, d INTEGER, time INTEGER,
        entry REAL, sl REAL, tp1 REAL, tp2 REAL, risk REAL, score REAL, why TEXT,
        status TEXT DEFAULT 'OPEN', r REAL, closed_time INTEGER, msg_id INTEGER,
        UNIQUE(symbol, d, time))""")
    con.row_factory = sqlite3.Row
    return con

def live_stats():
    with db() as con:
        rows = [dict(x) for x in con.execute("SELECT symbol, score, r FROM signals WHERE status!='OPEN'")]
        open_n = con.execute("SELECT COUNT(*) FROM signals WHERE status='OPEN'").fetchone()[0]
    rs = lambda sel: [x["r"] for x in rows if sel(x)]
    return {
        **summarize(rs(lambda x: True)), "open": open_n,
        "by_asset": {n: summarize(rs(lambda x, n=n: x["symbol"] == n)) for n in ASSETS
                     if rs(lambda x, n=n: x["symbol"] == n)},
        "by_score": {"low <40": summarize(rs(lambda x: x["score"] < 40)),
                     "mid 40-59": summarize(rs(lambda x: 40 <= x["score"] < 60)),
                     "high 60+": summarize(rs(lambda x: x["score"] >= 60))},
    }

# ------------------------------------------------------------------ telegram
def tg_send(text, reply_to=None):
    if not TOKEN or not CHANNEL:
        log.info("[dry-run] %s", text)
        return None
    payload = {"chat_id": CHANNEL, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()["result"]["message_id"]
    except Exception as e:
        log.error("Telegram send failed: %s", e)
        return None

def fmt_time(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def alert_text(sid, s, rank, total):
    a = ASSETS[s["symbol"]]
    px = lambda x: f"{x:,.{a['dec']}f}"
    side = "LONG 🟢" if s["d"] == 1 else "SHORT 🔴"
    chase = s["entry"] + s["d"] * CFG["max_chase_r"] * s["risk"]
    return (
        f"<b>#{s['symbol']} {side}</b> (signal #{sid})\n"
        f"⭐ Setup score <b>{s['score']:.0f}/100</b>, ranked #{rank} of {total} this scan "
        f"(a ranking, not a win probability)\n"
        f"🕒 30m candle close: {fmt_time(s['time'])}\n\n"
        f"Entry: <code>{px(s['entry'])}</code> (skip if price already past <code>{px(chase)}</code>)\n"
        f"Stop:  <code>{px(s['sl'])}</code>\n"
        f"TP1:   <code>{px(s['tp1'])}</code> (close half, stop to entry)\n"
        f"TP2:   <code>{px(s['tp2'])}</code>\n\n"
        f"Why: {s['why']}\n"
        f"Feed: {a['feed']}\n\n"
        f"Risk 0.5-1% of capital: size = (capital x risk%) / |entry - stop|. "
        f"Every signal and outcome is logged and posted. Not financial advice."
    )

def outcome_text(sid, s, status, r, ct):
    st = live_stats()
    icon = {"SL": "❌", "TP1_BE": "🟡", "TP2": "✅", "EXPIRED": "⏱️"}[status]
    return (f"{icon} <b>#{s['symbol']} signal #{sid} closed: {status}</b> ({r:+.2f}R, costs included)\n"
            f"Closed: {fmt_time(ct)}\n\n"
            f"📒 Record: {st['trades']} trades | win {st.get('win_rate', 0)}% | "
            f"total {st.get('total_r', 0)}R | PF {st.get('profit_factor')} | max DD {st.get('max_drawdown_r', 0)}R")

# ------------------------------------------------------------------ live cycle
_lock = threading.Lock()
last_scanned = {}

def settle_open(name, df30):
    with db() as con:
        for row in con.execute("SELECT * FROM signals WHERE symbol=? AND status='OPEN'", (name,)).fetchall():
            s = dict(row)
            s["cost_pct"] = ASSETS[name]["cost_pct"]
            res = resolve(s, df30[df30.open_time > row["time"]])
            if res:
                status, r, ct, _ = res
                con.execute("UPDATE signals SET status=?, r=?, closed_time=? WHERE id=?", (status, r, ct, row["id"]))
                con.commit()
                tg_send(outcome_text(row["id"], s, status, r, ct), reply_to=row["msg_id"])

def publish(cands):
    if not cands:
        return "no setups on newly closed candles"
    cands.sort(key=lambda s: s["score"], reverse=True)
    sent, used_groups, posted = 0, set(), []
    for rank, s in enumerate(cands, 1):
        grp = ASSETS[s["symbol"]]["group"]
        if sent >= CFG["max_per_cycle"] or grp in used_groups:
            continue
        with db() as con:
            open_syms = [x[0] for x in con.execute("SELECT symbol FROM signals WHERE status='OPEN'")]
            if any(ASSETS.get(x, {}).get("group") == grp for x in open_syms):
                continue  # correlated exposure already open
            try:
                cur = con.execute(
                    "INSERT INTO signals(symbol,d,time,entry,sl,tp1,tp2,risk,score,why) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (s["symbol"], s["d"], s["time"], s["entry"], s["sl"], s["tp1"], s["tp2"], s["risk"], s["score"], s["why"]))
            except sqlite3.IntegrityError:
                continue
            sid = cur.lastrowid
            con.execute("UPDATE signals SET msg_id=? WHERE id=?", (tg_send(alert_text(sid, s, rank, len(cands))), sid))
            con.commit()
        used_groups.add(grp)
        sent += 1
        posted.append(f"{s['symbol']}#{sid}({s['score']})")
    return f"posted {sent} of {len(cands)} candidates: {', '.join(posted) or 'none (group already open)'}"

def run_cycle():
    if not _lock.acquire(blocking=False):
        return "cycle already running"
    try:
        now = int(time.time() * 1000)
        expected = now // 1_800_000 * 1_800_000 - 1  # close_time of the latest 30m candle
        cands = []
        for name in ENABLED:
            if last_scanned.get(name, 0) >= expected:
                continue  # this candle already evaluated
            a = ASSETS[name]
            try:
                df4h, df30 = get_frames(name)
            except Exception:
                log.exception("data error %s", name)
                continue
            if df4h.empty or df30.empty:
                continue
            settle_open(name, df30)
            last = int(df30.close_time.iloc[-1])
            if last <= last_scanned.get(name, 0):
                continue  # feed hasn't published the new candle yet; retried on the next pass
            last_scanned[name] = last
            if now - last > 60 * 60_000:
                continue  # market closed or stale feed: never signal on an old candle
            f = build_frame(df4h, df30, a)
            s = evaluate(name, f, len(f) - 1, a)
            if s and s["score"] >= CFG["min_score"]:
                cands.append(s)
        return publish(cands)
    finally:
        _lock.release()

def worker():
    """Passes at +10s/+40s/+90s/+150s after each 30m close, so slow feeds are still caught quickly."""
    while True:
        boundary = (int(time.time() // 1800) + 1) * 1800
        for off in (10, 40, 90, 150):
            time.sleep(max(0, boundary + off - time.time()))
            try:
                log.info(run_cycle())
            except Exception:
                log.exception("cycle failed")

_started = False
def start_worker():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=worker, daemon=True).start()

# ------------------------------------------------------------------ web
@app.route("/")
def home(): return "Multi-asset ranked bot online", 200

@app.route("/scan")
def scan():
    try:
        return jsonify(status="ok", message=run_cycle())
    except Exception as e:
        log.exception("scan failed")
        return jsonify(status="error", error=str(e)), 500

@app.route("/stats")
def stats(): return jsonify(live_stats())

@app.route("/signals")
def signals():
    with db() as con:
        return jsonify([dict(x) for x in con.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 100")])

# ------------------------------------------------------------------ backtest (per asset)
def backtest(days=180):
    all_rs = []
    for name in ENABLED:
        a = ASSETS[name]
        df4h, df30 = get_frames(name, days)
        f = build_frame(df4h, df30, a)
        if f.empty:
            print(f"{name}: no data"); continue
        trades, i = [], 250
        while i < len(f) - 1:
            s = evaluate(name, f, i, a)
            if not s:
                i += 1; continue
            res = resolve(s, f.iloc[i + 1: i + 1 + CFG["max_hold"]])
            if not res:
                break
            trades.append((s["score"], res[1]))
            i += res[3] + 1
        rs = [t[1] for t in trades]
        all_rs += rs
        h = len(rs) // 2
        span = (f.close_time.iloc[-1] - f.close_time.iloc[250]) / 86_400_000
        print(f"\n{name}  ({span:.0f} days tested, {a['src']})")
        print("  all      ", summarize(rs))
        print("  1st half ", summarize(rs[:h]))
        print("  2nd half ", summarize(rs[h:]))
    print("\nCOMBINED", summarize(all_rs))
    print("\nTrust an asset only if BOTH halves are positive with a meaningful trade count (30+).")
    print("Yahoo 30m data is capped at ~60 days: use data/<ASSET>_30m.csv for a real test.")

if os.getenv("DISABLE_WORKER") != "1" and not (len(sys.argv) > 1 and sys.argv[1] == "backtest"):
    start_worker()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest(int(sys.argv[2]) if len(sys.argv) > 2 else 180)
    else:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))