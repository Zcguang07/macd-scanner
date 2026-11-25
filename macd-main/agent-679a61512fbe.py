import datetime
import re
import os
import time
import json
from typing import Dict, List
import numpy as np
import pandas as pd
import requests
import yfinance as yf

WEBHOOKS_ENV = os.environ.get("DISCORD_WEBHOOK", "")
DISCORD_WEBHOOKS = [url.strip() for url in WEBHOOKS_ENV.split(",") if url.strip()]

LOG_FILE = "sent_signals.json"

def load_sent_log() -> dict:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

REC_KEY = "__records__"   # 前瞻记录存放在同一个 json 里, 不需要新文件或改 workflow

def save_sent_log(log_dict: dict):
    # 设定过期时间，自动清理 30 天前的旧记录
    expire_seconds = 30 * 24 * 3600
    current_time = time.time()
    
    cleaned_log = {}
    for key, value in log_dict.items():
        if key == REC_KEY:
            continue
        if isinstance(value, bool):
            cleaned_log[key] = current_time
        elif isinstance(value, (int, float)) and (current_time - value) < expire_seconds:
            cleaned_log[key] = value

    # 前瞻记录需要保留到观察窗口结束 (周线约半年)，只按数量封顶，保留最近 400 条
    recs = log_dict.get(REC_KEY)
    if isinstance(recs, dict) and recs:
        items = sorted(recs.items(), key=lambda kv: str(kv[1].get("t", "")) if isinstance(kv[1], dict) else "")
        cleaned_log[REC_KEY] = dict(items[-400:])

    with open(LOG_FILE, "w") as f:
        json.dump(cleaned_log, f, indent=2)

def load_watchlist(filename="watchlist.txt") -> List[str]:
    if not os.path.exists(filename):
        print(f"未找到 {filename}，请创建并填入股票代码。")
        return []
    with open(filename, "r") as f:
        tickers = list(set([line.strip().upper() for line in f if line.strip()]))
        return sorted(tickers)

def calculate_macd_and_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["HIST"] = (df["DIF"] - df["DEA"]) * 2
    
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    df["EMA_250"] = df["Close"].ewm(span=250, adjust=False).mean()
    
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 核心抗插针处理：计算 K 线实体下沿与上沿
    df["Body_Low"] = df[["Open", "Close"]].min(axis=1)
    df["Body_High"] = df[["Open", "Close"]].max(axis=1)

    # 混合极值：常规时段 (RTH) 用真实影线，盘前盘后 (ETH) 用实体
    # 日线/周线本身就是 RTH 数据 (未开 prepost)，没有 is_rth 列时视为全部 RTH
    if "is_rth" in df.columns:
        df["Sig_Low"] = df["Low"].where(df["is_rth"], df["Body_Low"])
        df["Sig_High"] = df["High"].where(df["is_rth"], df["Body_High"])
    else:
        df["Sig_Low"] = df["Low"]
        df["Sig_High"] = df["High"]
    return df


# ===================== 周期参数 =====================
# 提示时机: 下跌(上涨)进行中就提示, 最晚比价格低点(高点)晚 3 根 K 线; 柱子翻到零轴另一侧后不再提示。
#   *_age    : 最早在低点(高点)出现后第几根开始提示 (0 = 不等待)
#   *_window : 在此之后最多再晚几根仍然提示  => 提示范围 = 低点(高点)后第 [age, age + window] 根, 现为 0~3 根
#   *_area   : 本轮红(绿)柱面积须小于上一轮的这个比例
#   *_gap_bars  : 1 = 不合并波段 (柱子过零轴就是新的一段)
#   *_wave_bars : 1 = 不过滤短波段
BASE_CFG = {
    'bottom_age': 0, 'bottom_window': 3, 'bottom_area': 0.5, 'min_gap_bars': 1, 'min_wave_bars': 1,
    'top_age': 0, 'top_window': 3, 'top_area': 0.5, 'top_gap_bars': 1, 'top_wave_bars': 1,
}
TF_CONFIG = {tf: dict(BASE_CFG) for tf in ('4h', '1d', '1wk')}

# 现价离本轮极值超过这么多个 ATR, 说明这轮 dip/top 已经过去, 不再推送
MAX_ATR_FROM_EXTREME = 3.0


def build_4h_session_bars(df_1h: pd.DataFrame) -> pd.DataFrame:
    """按美股交易时段切 4H K 线，避免盘前和盘中混进同一根 K 线：
    盘前 04:00-09:30 | 盘中1 09:30-13:30 | 盘中2 13:30-16:00 | 盘后 16:00-20:00 (美东时间)"""
    df = df_1h.copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    df.index = idx

    minutes = idx.hour * 60 + idx.minute
    day = idx.normalize()
    offset_min = np.select(
        [minutes < 570, minutes < 810, minutes < 960],
        [240, 570, 810],
        default=960,
    )
    bucket = day + pd.to_timedelta(offset_min, unit="m")
    is_rth = (minutes >= 570) & (minutes < 960)

    df["_bucket"] = bucket
    df["_is_rth"] = is_rth
    out = df.groupby("_bucket").agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"), is_rth=("_is_rth", "first"),
    ).dropna(subset=["Open", "High", "Low", "Close"])
    out.index.name = None
    out["is_rth"] = out["is_rth"].astype(bool)
    return out


def _extract_waves(df: pd.DataFrame, side: str, min_gap_bars: int, min_wave_bars: int) -> List[Dict]:
    """通用波段提取。side='bottom' 取绿柱，side='top' 取红柱。
    夹在两段同色柱之间、长度 < min_gap_bars 的反色小段会被合并掉。"""
    waves = []
    if len(df) < 20:
        return waves

    hist = df["HIST"].to_numpy()
    active = hist < 0 if side == "bottom" else hist > 0
    price_col = "Sig_Low" if side == "bottom" else "Sig_High"

    # 1) 切分连续同色段 [start, end, is_active]
    segs = []
    s = 0
    for i in range(1, len(active) + 1):
        if i == len(active) or active[i] != active[s]:
            segs.append([s, i - 1, bool(active[s])])
            s = i

    # 2) 合并：短反色段 + 两侧同色段 → 一个波段
    i = 1
    while i < len(segs) - 1:
        gap = segs[i]
        if (not gap[2]) and (gap[1] - gap[0] + 1) < min_gap_bars and segs[i - 1][2] and segs[i + 1][2]:
            segs[i - 1][1] = segs[i + 1][1]
            del segs[i:i + 2]
        else:
            i += 1

    # 3) 生成波段
    for s, e, is_active in segs:
        if not is_active:
            continue
        g = df.iloc[s:e + 1]
        g_active = g[active[s:e + 1]]
        if len(g_active) < min_wave_bars:
            continue
        if side == "bottom":
            extreme_dif = g["DIF"].min()
            if extreme_dif >= 0:
                continue
            extreme_price = g[price_col].min()
        else:
            extreme_dif = g["DIF"].max()
            if extreme_dif <= 0:
                continue
            extreme_price = g[price_col].max()
        waves.append({
            "area": float(g_active["HIST"].abs().sum()),
            "extreme_dif": float(extreme_dif),
            "extreme_price": float(extreme_price),
            "start_time": g.index[0],
            "end_time": g.index[-1],
            "end_idx": e,
            "extreme_idx": s + int(np.argmin(g[price_col].to_numpy()) if side == "bottom" else np.argmax(g[price_col].to_numpy())),
        })
    return waves


def get_bottom_waves(df: pd.DataFrame, timeframe: str) -> List[Dict]:
    """底背离绿柱波段：合并下跌途中的短暂红柱反抽，过滤过短波段 (更迟钝)"""
    cfg = TF_CONFIG[timeframe]
    return _extract_waves(df, "bottom", cfg["min_gap_bars"], cfg["min_wave_bars"])


def get_top_waves(df: pd.DataFrame, timeframe: str) -> List[Dict]:
    """顶背离红柱波段：参数独立于底背离 (回测显示等 1 根确认后误报明显更少)"""
    cfg = TF_CONFIG[timeframe]
    return _extract_waves(df, "top", cfg["top_gap_bars"], cfg["top_wave_bars"])


# ===================== 辅助信息 (环境 / 量能 / 位置 / 财报) =====================
# 这些只作为卡片上的参考信息, 未经回测验证; 任何一项取不到都不影响背离提醒本身
_CTX_CACHE = {}


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_daily(ticker: str):
    """日线数据 (每次运行内缓存, 仅在触发信号时才会调用)"""
    key = ("D", ticker)
    if key not in _CTX_CACHE:
        d = None
        try:
            raw = yf.download(ticker, period="2y", interval="1d", progress=False)
            if raw is not None and not raw.empty:
                d = _flatten(raw).dropna(subset=["Close"])
        except Exception:
            d = None
        _CTX_CACHE[key] = d
    return _CTX_CACHE[key]


def trend_context(ticker: str):
    d = fetch_daily(ticker)
    if d is None or len(d) < 60:
        return None
    c = d["Close"].astype(float)
    last = float(c.iloc[-1])
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    out = {"close": last}
    s200 = float(sma200.iloc[-1]) if len(c) >= 200 and pd.notna(sma200.iloc[-1]) else None
    s50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    slope = None
    if len(c) >= 221 and pd.notna(sma200.iloc[-21]):
        slope = float(sma200.iloc[-1] / sma200.iloc[-21] - 1)
    out["dist200"] = (last / s200 - 1) if s200 else None
    out["slope200"] = slope
    if s200 is None or slope is None:
        state, label = None, "上市不足一年"
    elif last > s200 and slope > 0:
        state, label = ("pullback", "上升趋势中的回调") if (s50 and last < s50) else ("uptrend", "上升趋势")
    elif last > s200:
        state, label = "weakening", "趋势转弱 (200 日线走平/下行)"
    elif slope <= 0:
        state, label = "downtrend", "下降趋势 (200 日线下方且向下)"
    else:
        state, label = "below_rising", "位于 200 日线下方 (200 日线仍上行)"
    out["state"], out["label"] = state, label
    hi60 = float(d["High"].astype(float).tail(60).max()); lo60 = float(d["Low"].astype(float).tail(60).min())
    hi252 = float(d["High"].astype(float).tail(252).max())
    out["dd60"] = last / hi60 - 1          # 距 60 日高点 (<=0)
    out["dd252"] = last / hi252 - 1        # 距 52 周高点 (<=0)
    out["runup60"] = last / lo60 - 1       # 较 60 日低点 (>=0)
    return out


def market_context():
    res = []
    for tk in ("SPY", "QQQ"):
        d = fetch_daily(tk)
        if d is None or len(d) < 30:
            continue
        c = d["Close"].astype(float)
        s200 = c.rolling(200).mean().iloc[-1]
        above = bool(c.iloc[-1] > s200) if pd.notna(s200) else None
        ret20 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else None
        res.append({"tk": tk, "above200": above, "ret20": ret20})
    return res


def market_is_weak(mkt) -> bool:
    return any((m["above200"] is False) or (m["ret20"] is not None and m["ret20"] <= -0.07) for m in (mkt or []))


def atr_value(df, n: int = 14):
    c = df["Close"].astype(float); h = df["High"].astype(float); l = df["Low"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.rolling(n, min_periods=5).mean().iloc[-1]
    return float(v) if pd.notna(v) else None


def is_stale(side, price, p2, atr):
    """底背离: 现价高于低点太多; 顶背离: 现价低于高点太多"""
    if not atr or atr <= 0:
        return False
    gap = (price - p2) if side == "bottom" else (p2 - price)
    return gap > MAX_ATR_FROM_EXTREME * atr


def extreme_bar_info(df, wave, side):
    """本轮极值所在 K 线: 成交量倍数 (相对同类时段前 20 根均量)、成交额、影线占比"""
    seg = df.loc[wave["start_time"]:wave["end_time"]]
    col = "Sig_Low" if side == "bottom" else "Sig_High"
    ts = seg[col].idxmin() if side == "bottom" else seg[col].idxmax()
    i = df.index.get_loc(ts)
    row = df.iloc[i]
    vol = df["Volume"].astype(float)
    if "is_rth" in df.columns:
        prev = vol.iloc[:i][df["is_rth"].iloc[:i] == row["is_rth"]].tail(20)
    else:
        prev = vol.iloc[max(0, i - 20):i]
    avg = float(prev.mean()) if len(prev) >= 5 and prev.mean() > 0 else None
    hi, lo = float(row["High"]), float(row["Low"])
    rng = hi - lo
    if side == "bottom":
        shadow = (float(row["Body_Low"]) - lo) / rng if rng > 0 else 0.0
    else:
        shadow = (hi - float(row["Body_High"])) / rng if rng > 0 else 0.0
    session = None
    if "is_rth" in df.columns:
        session = "盘中" if bool(row["is_rth"]) else "盘前盘后"
    return {
        "time": ts, "age": (df.index[-1] - ts), "vol_mult": (float(row["Volume"]) / avg) if (avg and row["Volume"] > 0) else None,
        "money": (float(row["Volume"]) * (hi + lo) / 2) if row["Volume"] > 0 else None, "shadow": float(shadow), "session": session,
    }


def _to_date(x):
    try:
        if isinstance(x, datetime.datetime): return x.date()
        if isinstance(x, datetime.date): return x
        return pd.Timestamp(x).date()
    except Exception:
        return None


def earnings_context(ticker: str):
    key = ("E", ticker)
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    today = datetime.date.today()
    nxt = last = None
    try:
        t = yf.Ticker(ticker)
        try:
            cal = t.calendar
            dates = []
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or []
                dates = list(ed) if isinstance(ed, (list, tuple)) else [ed]
            elif cal is not None and hasattr(cal, "loc"):
                dates = list(cal.loc["Earnings Date"])
            ups = sorted(d for d in (_to_date(x) for x in dates) if d and d >= today)
            nxt = ups[0] if ups else None
        except Exception:
            pass
        try:
            ed = t.get_earnings_dates(limit=8)
            if ed is not None and len(ed):
                past = [d for d in (_to_date(x) for x in ed.index) if d and d <= today]
                last = max(past) if past else None
        except Exception:
            pass
    except Exception:
        pass
    _CTX_CACHE[key] = {"next": nxt, "last": last}
    return _CTX_CACHE[key]


def build_extras(ticker, tf, df, wave, side):
    ex = {}
    for name, fn in (("bar", lambda: extreme_bar_info(df, wave, side)), ("trend", lambda: trend_context(ticker)),
                     ("market", market_context), ("earn", lambda: earnings_context(ticker)), ("atr", lambda: atr_value(df))):
        try:
            ex[name] = fn()
        except Exception:
            ex[name] = None
    return ex


def _fmt_money(x):
    if x >= 1e8: return f"{x/1e8:.1f} 亿美元"
    if x >= 1e4: return f"{x/1e4:.0f} 万美元"
    return f"{x:.0f} 美元"


def _fmt_age(delta):
    try:
        if delta.days >= 1:
            return f"{delta.days}天前"
        return f"{int(delta.total_seconds() // 3600)}小时前"
    except Exception:
        return ""


def _fmt_bar_time(ts, tf):
    try:
        return ts.strftime("%m-%d %H:%M") if tf == "4h" else ts.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def _earn_text(earn):
    today = datetime.date.today()
    if not earn:
        return None
    parts = []
    if earn.get("next"):
        parts.append(f"财报 {earn['next']:%m-%d} ({(earn['next'] - today).days}天后)")
    if earn.get("last") and 0 <= (today - earn["last"]).days <= 3:
        parts.append(f"刚发财报 ({(today - earn['last']).days}天前)")
    return " · ".join(parts) if parts else None


def build_warnings(signal_type, ex, tf):
    """只列几个字的风险点, 没有就返回空列表"""
    w = []
    today = datetime.date.today()
    earn = ex.get("earn") or {}
    if earn.get("next") and 0 <= (earn["next"] - today).days <= 7:
        w.append(f"财报{(earn['next'] - today).days}天后")
    if signal_type == "bottom" and earn.get("last") and 0 <= (today - earn["last"]).days <= 3:
        w.append(f"刚发财报{(today - earn['last']).days}天")
    tr, bar, mkt = ex.get("trend"), ex.get("bar"), ex.get("market")
    vm = bar.get("vol_mult") if bar else None
    if signal_type == "bottom":
        if tr and tr.get("state") == "downtrend":
            w.append("下降趋势")
        if market_is_weak(mkt):
            w.append("大盘走弱")
        if tr and tr.get("dd60") is not None and tr["dd60"] > -0.08:
            w.append(f"回调仅{abs(tr['dd60'])*100:.0f}%")
        if vm is not None and vm < 0.7:
            w.append(f"低点缩量{vm:.1f}×")
    else:
        if tr and tr.get("runup60") is not None and tr["runup60"] >= 0.40:
            w.append(f"60日已涨{tr['runup60']*100:.0f}%")
        if vm is not None and vm >= 2.0:
            w.append(f"高点放量{vm:.1f}×")
    return w


# ===================== 前瞻记录: 每次提醒后记下价格，窗口到期后自动补上实际走势 =====================
# 窗口与 backtest.py 一致 (K 线根数)
HORIZONS = {'4h': [20, 40], '1d': [21, 42], '1wk': [13, 26]}
HORIZON_TEXT = {'4h': '两周', '1d': '两个月', '1wk': '半年'}


def record_signal(sent_log, signal_id, ticker, tf, kind, df, price, p2):
    try:
        recs = sent_log.setdefault(REC_KEY, {})
        recs[signal_id] = {"ticker": ticker, "tf": tf, "kind": kind, "t": df.index[-1].isoformat(),
                           "px": round(float(price), 4), "p2": round(float(p2), 4), "res": {}}
    except Exception as e:
        print(f"前瞻记录写入失败(不影响提醒): {e}")


def update_records(sent_log, ticker, tf, df):
    try:
        recs = sent_log.get(REC_KEY)
        if not isinstance(recs, dict):
            return
        hs = HORIZONS[tf]
        n = len(df)
        hi = df["High"].to_numpy(dtype=float); lo = df["Low"].to_numpy(dtype=float); c = df["Close"].to_numpy(dtype=float)
        for rec in recs.values():
            if not isinstance(rec, dict) or rec.get("ticker") != ticker or rec.get("tf") != tf:
                continue
            res = rec.setdefault("res", {})
            if all(str(h) in res for h in hs):
                continue
            ts = pd.Timestamp(rec["t"])
            if df.index.tz is not None:
                ts = ts.tz_localize(df.index.tz) if ts.tzinfo is None else ts.tz_convert(df.index.tz)
            elif ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            p = int(df.index.get_indexer([ts])[0])
            if p < 0:
                continue
            entry = float(rec["px"])
            for h in hs:
                end = p + h
                if str(h) in res or end > n - 2:      # 最后一根可能还没走完，不用
                    continue
                res[str(h)] = [round(float(c[end] / entry - 1), 4), round(float(lo[p + 1:end + 1].min() / entry - 1), 4),
                               round(float(hi[p + 1:end + 1].max() / entry - 1), 4)]
    except Exception as e:
        print(f"前瞻记录更新失败(不影响提醒): {e}")


def build_track_text(sent_log) -> str:
    """只显示已有到期结果的组合; 全部待到期则返回空串 (汇总卡片不显示这一栏)"""
    recs = sent_log.get(REC_KEY)
    if not isinstance(recs, dict) or not recs:
        return ""
    groups = {}
    for r in recs.values():
        if isinstance(r, dict) and r.get("kind") and r.get("tf") in HORIZONS:
            groups.setdefault((r["kind"], r["tf"]), []).append(r)
    lines = []
    for (kind, tf), lst in sorted(groups.items(), key=lambda kv: (kv[0][0], ['4h', '1d', '1wk'].index(kv[0][1]))):
        hl = str(HORIZONS[tf][-1])
        done = [r for r in lst if hl in r.get("res", {})]
        if not done:
            continue
        n = len(done)
        dn = sum(1 for r in done if r["res"][hl][1] <= -0.12)
        up = sum(1 for r in done if r["res"][hl][0] > 0)
        lines.append(f"{tf}{'底' if kind == 'bottom' else '顶'} {HORIZON_TEXT[tf]}后 {n}个: 曾跌≥12% {dn/n:.0%} · 期末上涨 {up/n:.0%}")
    return "\n".join(lines[:6])


def send_discord_card(data: dict, signal_type: str):
    if not DISCORD_WEBHOOKS: return

    tf = data['timeframe']
    price = data['current_price']
    ema20, sma200, rsi = data['ema_20'], data['sma_200'], data['rsi_14']
    P1, P2 = data['P1'], data['P2']
    ex = data.get('extras') or {}
    bottom = signal_type == "bottom"

    dist = (price / P2 - 1) * 100
    atr = ex.get('atr')
    atr_txt = f" · {abs(price - P2) / atr:.1f} ATR" if atr else ""
    warns = build_warnings(signal_type, ex, tf)

    title = f"{'🟢' if bottom else '🔴'} {data['ticker']} · {tf} {'底背离' if bottom else '顶背离'}"
    color = (0x3498DB if tf == '4h' else (0x9B59B6 if tf == '1d' else 0x2ECC71)) if bottom else 0xE74C3C
    desc = f"`${price:.2f}`" + (f"　⚠️ {' · '.join(warns)}" if warns else "")

    fields = [
        {"name": "背离", "value": f"价格 `${P1:.2f}` → `${P2:.2f}`\nDIF `{data['A1']:.2f}` → `{data['A2']:.2f}`\n面积 {data['ratio']:.0%}", "inline": True},
        {"name": "关键位", "value": f"{'低点' if bottom else '高点'} `${P2:.2f}` ({dist:+.1f}%{atr_txt})\nEMA20 `${ema20:.2f}` · SMA200 `${sma200:.2f}`\nRSI {rsi:.0f}", "inline": True},
    ]

    tr, bar, mkt, earn = ex.get('trend'), ex.get('bar'), ex.get('market'), ex.get('earn')
    env = []
    if tr:
        env.append(tr['label'] + (f" ({tr['dist200']*100:+.1f}%)" if tr.get('dist200') is not None else ""))
    if mkt:
        env.append(" · ".join(f"{m['tk']} {'' if m['above200'] is None else ('站上' if m['above200'] else '跌破')}200日线"
                              + ("" if m['ret20'] is None else f" {m['ret20']*100:+.1f}%") for m in mkt))
    if env:
        fields.append({"name": "环境", "value": "\n".join(env)[:1000], "inline": True})

    if bar:
        sess = f" {bar['session']}" if bar.get('session') else ""
        age = f" · {_fmt_age(bar['age'])}" if bar.get('age') is not None else ""
        lines = [f"{_fmt_bar_time(bar['time'], tf)}{sess}{age}"]
        if bar.get('vol_mult') is not None and bar.get('money'):
            lines.append(f"量 {bar['vol_mult']:.1f}× · {_fmt_money(bar['money'])}")
        lines.append(f"{'下影' if bottom else '上影'} {bar['shadow']*100:.0f}%")
        fields.append({"name": "低点K线" if bottom else "高点K线", "value": "\n".join(lines), "inline": True})

    pos = []
    if tr:
        pos.append((f"距60日高 {tr['dd60']*100:.0f}%" if bottom else f"较60日低 +{tr['runup60']*100:.0f}%")
                   + f" · 距52周高 {tr['dd252']*100:.0f}%")
    et = _earn_text(earn)
    if et:
        pos.append(et)
    if pos:
        fields.append({"name": "位置", "value": "\n".join(pos)[:1000], "inline": True})

    embed = {
        "title": title,
        "color": color,
        "description": desc,
        "fields": fields,
        "footer": {"text": "MACD Bi-directional Agent"},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    payload = {"username": "Signal Agent", "embeds": [embed]}
    for webhook_url in DISCORD_WEBHOOKS:
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception:
            pass

def build_summary_desc(triggered_list: list) -> str:
    """按 底/顶 各一行; 只有同一标的多个周期同时触发时才标 ⭐共振"""
    groups = {"底": {}, "顶": {}}
    for s in triggered_list:
        m = re.match(r"^(.+)\((\w+?)(底|顶)\)$", s)
        if not m:
            return f"共 `{len(triggered_list)}` 个新提醒：`{', '.join(triggered_list)}`"
        groups[m.group(3)].setdefault(m.group(1), []).append(m.group(2))

    def fmt(g):
        return "、".join(f"**{tk}** ({', '.join(tfs)})" + (" ⭐共振" if len(tfs) >= 2 else "") for tk, tfs in g.items())

    parts = [f"共 `{len(triggered_list)}` 个新提醒"]
    if groups["底"]:
        parts.append("🟢 底背离: " + fmt(groups["底"]))
    if groups["顶"]:
        parts.append("🔴 顶背离: " + fmt(groups["顶"]))
    return "\n".join(parts)


def send_summary_card(watchlist: list, triggered_list: list, track_text: str = ""):
    # 绝对静默拦截
    if not DISCORD_WEBHOOKS or not triggered_list: 
        return

    scanned_count = len(watchlist)
    tickers_str = ", ".join(watchlist)
    safe_value_str = "```text\n" + tickers_str + "\n```\n"

    desc = build_summary_desc(triggered_list)
    color = 0x2ECC71

    embed = {
        "title": "📡 量化双向扫描任务报告",
        "description": desc,
        "color": color,
        "fields": [
            {
                "name": f"当前监控池 ({scanned_count} 只)",
                "value": safe_value_str,
                "inline": False
            }
        ],
        "footer": {"text": "GitHub Actions Auto-Run"},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    
    if track_text:
        embed["fields"].append({"name": "📒 前瞻记录", "value": track_text[:1000], "inline": False})

    payload = {"username": "System Monitor", "embeds": [embed]}
    for webhook_url in DISCORD_WEBHOOKS:
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except Exception:
            pass

def process_timeframe(ticker: str, timeframe: str, sent_log: dict) -> List[str]:
    """返回本次触发的信号字符串列表，例如 ['AAPL(4h底)', 'MSFT(1d顶)']"""
    triggered = []
    try:
        if timeframe == '4h':
            df = yf.download(ticker, period="60d", interval="1h", prepost=True, progress=False)
            if df.empty: return triggered
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = build_4h_session_bars(df)
        elif timeframe == '1d':
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty: return triggered
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        elif timeframe == '1wk':
            df = yf.download(ticker, period="5y", interval="1wk", progress=False)
            if df.empty: return triggered
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df = calculate_macd_and_ema(df)
        latest_close = float(df["Close"].iloc[-1])
        cfg = TF_CONFIG[timeframe]
        last_idx = len(df) - 1

        update_records(sent_log, ticker, timeframe, df)   # 前瞻记录: 补全已到期窗口
        atr_now = atr_value(df)

        def card_data(P1, P2, A1, A2, B1, B2, wave, side):
            return {
                "ticker": ticker, "timeframe": timeframe, "current_price": latest_close,
                "P1": P1, "P2": P2, "A1": A1, "A2": A2, "B1": B1, "B2": B2, "ratio": B2 / B1,
                "ema_20": float(df["EMA_20"].iloc[-1]), "sma_200": float(df["SMA_200"].iloc[-1]),
                "rsi_14": float(df["RSI_14"].iloc[-1]),
                "extras": build_extras(ticker, timeframe, df, wave, side),
            }

        # =============== 1. 底背离检测: 下跌进行中, 最晚比价格低点晚 3 根 ===============
        # 条件: 柱子还在零轴下方 (翻正后不再提示); 距本轮价格低点不超过 3 根 K 线;
        #       价格比上一轮更低 (P2 < P1); 红柱回升: DIF 低点比上一轮高 (A2 > A1), 且红柱面积小于上一轮的 50%
        bottom_waves = get_bottom_waves(df, timeframe)
        if len(bottom_waves) >= 2:
            w1, w2 = bottom_waves[-2], bottom_waves[-1]
            in_progress = (last_idx - w2["end_idx"]) == 0 and float(df["HIST"].iloc[-1]) < 0
            low_age = last_idx - w2["extreme_idx"]

            if in_progress and cfg["bottom_age"] <= low_age <= cfg["bottom_age"] + cfg["bottom_window"]:
                P1, A1, B1 = w1["extreme_price"], w1["extreme_dif"], w1["area"]
                P2, A2, B2 = w2["extreme_price"], w2["extreme_dif"], w2["area"]

                core_ok = P2 < P1 and A2 > A1 and B2 < (cfg["bottom_area"] * B1)
                if core_ok and is_stale("bottom", latest_close, P2, atr_now):
                    print(f"[{ticker}] {timeframe} 底背离已过期 (现价高于低点 {(latest_close - P2) / atr_now:.1f} ATR)，不推送")
                    core_ok = False
                if core_ok:
                    signal_date = w2["start_time"].strftime("%Y-%m-%d")
                    signal_id = f"{ticker}_{timeframe}_{signal_date}_bottom"

                    if signal_id not in sent_log:
                        send_discord_card(card_data(P1, P2, A1, A2, B1, B2, w2, "bottom"), "bottom")
                        record_signal(sent_log, signal_id, ticker, timeframe, "bottom", df, latest_close, P2)
                        sent_log[signal_id] = time.time()
                        triggered.append(f"{ticker}({timeframe}底)")
                        print(f"[{ticker}] 触发 {timeframe} 级别 底背离！")

        # =============== 2. 顶背离检测: 上涨进行中, 最晚比价格高点晚 3 根 ===============
        top_waves = get_top_waves(df, timeframe)
        if len(top_waves) >= 2:
            tw1, tw2 = top_waves[-2], top_waves[-1]
            in_progress = (last_idx - tw2["end_idx"]) == 0 and float(df["HIST"].iloc[-1]) > 0
            high_age = last_idx - tw2["extreme_idx"]

            if in_progress and cfg["top_age"] <= high_age <= cfg["top_age"] + cfg["top_window"]:
                P1, A1, B1 = tw1["extreme_price"], tw1["extreme_dif"], tw1["area"]
                P2, A2, B2 = tw2["extreme_price"], tw2["extreme_dif"], tw2["area"]

                core_ok = P2 > P1 and A2 < A1 and B2 < (cfg["top_area"] * B1)
                if core_ok and is_stale("top", latest_close, P2, atr_now):
                    print(f"[{ticker}] {timeframe} 顶背离已过期 (现价低于高点 {(P2 - latest_close) / atr_now:.1f} ATR)，不推送")
                    core_ok = False
                if core_ok:
                    signal_date = tw2["start_time"].strftime("%Y-%m-%d")
                    signal_id = f"{ticker}_{timeframe}_{signal_date}_top"

                    if signal_id not in sent_log:
                        send_discord_card(card_data(P1, P2, A1, A2, B1, B2, tw2, "top"), "top")
                        record_signal(sent_log, signal_id, ticker, timeframe, "top", df, latest_close, P2)
                        sent_log[signal_id] = time.time()
                        triggered.append(f"{ticker}({timeframe}顶)")
                        print(f"[{ticker}] 触发 {timeframe} 级别 顶背离！")

    except Exception as e:
        print(f"处理 {ticker} {timeframe} 时报错: {e}")
    return triggered

def main():
    watchlist = load_watchlist()
    if not watchlist: return
    print(f"[{datetime.datetime.now()}] 启动多周期顶底双向扫描...")
    
    sent_log = load_sent_log()
    all_triggered_signals = []

    for ticker in watchlist:
        signals = process_timeframe(ticker, '4h', sent_log)
        all_triggered_signals.extend(signals)
        
        signals = process_timeframe(ticker, '1d', sent_log)
        all_triggered_signals.extend(signals)
        
        signals = process_timeframe(ticker, '1wk', sent_log)
        all_triggered_signals.extend(signals)
        
        time.sleep(1)

    save_sent_log(sent_log)
    send_summary_card(watchlist, all_triggered_signals, build_track_text(sent_log))
    print(f"[{datetime.datetime.now()}] 扫描完成。")

if __name__ == "__main__":
    main()
