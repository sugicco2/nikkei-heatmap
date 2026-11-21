import json
import math
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

# 重複を除いた全銘柄コードを一括ダウンロード（高速化）
all_codes = sorted(list(set(df_nk["code"].tolist() + df_tp["code"].tolist())))
print(f"合計 {len(all_codes)} 銘柄の株価データを取得中...")
data = yf.download(all_codes, period="2y", group_by="ticker", auto_adjust=True, threads=True)

# TOPIX用の最新時価総額を並列で高速取得（失敗時は基準株式数から自動補完）
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

# 2. 各銘柄の現在値と騰落率を共通計算
metrics = {}
for code in all_codes:
    try:
        closes = data[code]["Close"].dropna()
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0].dropna()
        if len(closes) < 2:
            continue

        latest = float(closes.iloc[-1])
        if not math.isfinite(latest) or latest <= 0:
            continue

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

        metrics[code] = {
            "price": round(latest, 1),
            "raw_price": latest,
            "1D": d1,
            "1W": w1,
            "1M": m1,
            "YTD": ytd,
            "1Y": y1
        }
    except Exception as e:
        print(f"スキップ ({code}): {e}")

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
            # 取得できた最新時価総額を優先し、未取得時のみ株式数×株価で計算
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

    return [{"name": sec, "children": ch} for sec, ch in sectors.items()]

output = {
    "updated_at": now_jst.strftime("%m/%d %H:%M"),
    "indices": {
        "nikkei225": build_index_data(df_nk, "nikkei"),
        "topix": build_index_data(df_tp, "topix")
    }
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

print("日経225 + TOPIX100 のデータ出力が完了しました！")
