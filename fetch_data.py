import json
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf

# 1. 両方の銘柄リストを読み込む
df_nk = pd.read_csv("tickers.csv").dropna(subset=["code", "name", "sector", "factor"])
df_nk["factor"] = pd.to_numeric(df_nk["factor"], errors="coerce")
df_nk = df_nk.dropna(subset=["factor"]).drop_duplicates(subset=["code"])
df_nk["code"] = df_nk["code"].astype(str).str.strip()

df_tp = pd.read_csv("topix_tickers.csv").dropna(subset=["code", "name", "sector", "float_ratio", "est_shares"])
df_tp["float_ratio"] = pd.to_numeric(df_tp["float_ratio"], errors="coerce")
df_tp["est_shares"] = pd.to_numeric(df_tp["est_shares"], errors="coerce")
df_tp = df_tp.dropna(subset=["float_ratio", "est_shares"]).drop_duplicates(subset=["code"])
df_tp["code"] = df_tp["code"].astype(str).str.strip()

# 指数用ティッカーも個別銘柄と一緒に一括ダウンロード
index_codes = ["^N225", "^TOPX", "1306.T"]
stock_codes = sorted(list(set(df_nk["code"].tolist() + df_tp["code"].tolist())))
all_codes = sorted(list(set(stock_codes + index_codes)))

print(f"合計 {len(all_codes)} 銘柄の株価・指数データを取得中...")
data = yf.download(all_codes, period="2y", group_by="ticker", auto_adjust=True, threads=True)

# TOPIX用の最新時価総額を並列取得
def get_market_cap(code):
    try:
        mc = yf.Ticker(code).fast_info.get("marketCap")
        if mc and math.isfinite(float(mc)) and float(mc) > 0:
            return code, float(mc)
    except Exception:
        pass
    return code, None

market_caps = {}
with ThreadPoolExecutor(max_workers=10) as executor:
    for code, mc in executor.map(get_market_cap, df_tp["code"].tolist()):
        if mc is not None:
            market_caps[code] = mc

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)
current_year = now_jst.year

def extract_closes(download_data, code):
    try:
        if isinstance(download_data.columns, pd.MultiIndex):
            if code not in download_data.columns.get_level_values(0):
                return None
            s = download_data[code]["Close"].dropna()
        else:
            s = download_data["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0].dropna()
        if len(s) < 2:
            return None
        return s
    except Exception:
        return None

def calc_period_returns(closes):
    latest = float(closes.iloc[-1])

    def calc_change(idx_offset):
        if len(closes) > idx_offset:
            prev = float(closes.iloc[-(idx_offset + 1)])
            if math.isfinite(prev) and prev > 0:
                val = round(((latest - prev) / prev) * 100, 2)
                return val if math.isfinite(val) else 0.0
        return 0.0

    d1 = calc_change(1)
    w1 = calc_change(5)
    m1 = calc_change(21)
    y1 = calc_change(245)

    prev_year_closes = closes[closes.index.year < current_year]
    if len(prev_year_closes) > 0:
        ytd_base = float(prev_year_closes.iloc[-1])
        if math.isfinite(ytd_base) and ytd_base > 0:
            ytd = round(((latest - ytd_base) / ytd_base) * 100, 2)
            if not math.isfinite(ytd):
                ytd = d1
        else:
            ytd = d1
    else:
        ytd = d1

    return {
        "latest": latest,
        "1D": d1,
        "1W": w1,
        "1M": m1,
        "YTD": ytd,
        "1Y": y1
    }

# 2. 各銘柄の現在値と騰落率を計算
metrics = {}
for code in stock_codes:
    closes = extract_closes(data, code)
    if closes is None:
        continue
    latest = float(closes.iloc[-1])
    if not math.isfinite(latest) or latest <= 0:
        continue

    res = calc_period_returns(closes)
    metrics[code] = {
        "price": round(latest, 1),
        "raw_price": latest,
        "1D": res["1D"],
        "1W": res["1W"],
        "1M": res["1M"],
        "YTD": res["YTD"],
        "1Y": res["1Y"]
    }

def build_index_data(df, mode):
    items = []
    total_val = 0.0

    for _, row in df.iterrows():
        code = row["code"]
        if code not in metrics:
            continue
        m = metrics[code]

        if mode == "nikkei":
            factor = float(row["factor"])
            if factor <= 0:
                continue
            val = m["raw_price"] * factor
        else:
            float_ratio = float(row["float_ratio"])
            est_shares = float(row["est_shares"])
            mc = market_caps.get(code, m["raw_price"] * est_shares * 1_000_000)
            val = mc * float_ratio

        if not math.isfinite(val) or val <= 0:
            continue

        total_val += val
        items.append({
            "code": code.replace(".T", ""),
            "name": str(row["name"]).strip(),
            "sector": str(row["sector"]).strip(),
            "val": val,
            "m": m
        })

    # 指数データが外部取得できなかった場合のフォールバック用加重平均騰落率
    weighted_ret = {p: 0.0 for p in ["1D", "1W", "1M", "YTD", "1Y"]}
    if total_val > 0:
        for it in items:
            w = it["val"] / total_val
            for p in weighted_ret:
                weighted_ret[p] += it["m"][p] * w
        for p in weighted_ret:
            weighted_ret[p] = round(weighted_ret[p], 2)

    sectors = {}
    for it in items:
        sec = it["sector"]
        if sec not in sectors:
            sectors[sec] = []
        w_pct = round((it["val"] / total_val) * 100, 2) if total_val > 0 else 0.1
        if w_pct <= 0:
            w_pct = 0.05

        m = it["m"]
        sectors[sec].append({
            "code": it["code"],
            "name": it["name"],
            "price": m["price"],
            "weight": w_pct,
            "1D": m["1D"],
            "1W": m["1W"],
            "1M": m["1M"],
            "YTD": m["YTD"],
            "1Y": m["1Y"]
        })

    return [{"name": sec, "children": ch} for sec, ch in sectors.items()], weighted_ret

nk_sectors, nk_fallback_ret = build_index_data(df_nk, "nikkei")
tp_sectors, tp_fallback_ret = build_index_data(df_tp, "topix")

# 3. 日経平均指数 & TOPIX指数のサマリーを作成
index_summary = {}

# 日経平均株価指数 (^N225)
nk_closes = extract_closes(data, "^N225")
if nk_closes is not None:
    res = calc_period_returns(nk_closes)
    index_summary["nikkei225"] = {
        "value": round(res["latest"], 2),
        "1D": res["1D"], "1W": res["1W"], "1M": res["1M"], "YTD": res["YTD"], "1Y": res["1Y"]
    }
else:
    index_summary["nikkei225"] = {"value": 0, **nk_fallback_ret}

# TOPIX指数 (^TOPX または 1306.T + Yahoo!ファイナンス)
tp_closes = extract_closes(data, "^TOPX")
if tp_closes is not None:
    res = calc_period_returns(tp_closes)
    index_summary["topix"] = {
        "value": round(res["latest"], 2),
        "1D": res["1D"], "1W": res["1W"], "1M": res["1M"], "YTD": res["YTD"], "1Y": res["1Y"]
    }
else:
    etf_closes = extract_closes(data, "1306.T")
    res = calc_period_returns(etf_closes) if etf_closes is not None else {"latest": 0, **tp_fallback_ret}
    tp_val = round(res["latest"] * 0.955, 2) if res["latest"] > 0 else 0

    try:
        req = urllib.request.Request(
            "https://finance.yahoo.co.jp/quote/998405.T",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            m = re.search(r'"price":"([0-9,]+\.[0-9]+)"', html)
            if m:
                tp_val = float(m.group(1).replace(",", ""))
    except Exception:
        pass

    index_summary["topix"] = {
        "value": round(tp_val, 2),
        "1D": res["1D"], "1W": res["1W"], "1M": res["1M"], "YTD": res["YTD"], "1Y": res["1Y"]
    }

output = {
    "updated_at": now_jst.strftime("%m/%d %H:%M"),
    "index_summary": index_summary,
    "indices": {
        "nikkei225": nk_sectors,
        "topix": tp_sectors
    }
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

print("指数データ + 日経225 + TOPIX100 の出力が完了しました！")
