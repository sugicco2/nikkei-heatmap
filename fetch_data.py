import json
import math
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta

# 1. 銘柄リストを読み込み、空欄や不正な行を自動で除外する
df_tickers = pd.read_csv("tickers.csv")
df_tickers = df_tickers.dropna(subset=["code", "name", "sector", "factor"])
df_tickers["factor"] = pd.to_numeric(df_tickers["factor"], errors="coerce")
df_tickers = df_tickers.dropna(subset=["factor"]).drop_duplicates(subset=["code"])

codes = [str(c).strip() for c in df_tickers["code"].tolist()]

# 2. 過去2年分の株価を一括ダウンロード
print("株価データを取得中...")
data = yf.download(codes, period="2y", group_by="ticker", auto_adjust=True, threads=True)

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)
current_year = now_jst.year

raw_items = []
total_adjusted_price = 0.0

for _, row in df_tickers.iterrows():
    code = str(row["code"]).strip()
    name = str(row["name"]).strip()
    sector = str(row["sector"]).strip()
    factor = float(row["factor"])

    if not math.isfinite(factor) or factor <= 0:
        continue

    try:
        if len(codes) == 1:
            closes = data["Close"].dropna()
        else:
            closes = data[code]["Close"].dropna()

        # Series以外（DataFrame等）になった場合の安全処理
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0].dropna()

        if len(closes) < 2:
            continue

        latest = float(closes.iloc[-1])
        if not math.isfinite(latest) or latest <= 0:
            continue

        adjusted_price = latest * factor
        if not math.isfinite(adjusted_price):
            continue

        total_adjusted_price += adjusted_price

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

        raw_items.append({
            "code": code.replace(".T", ""),
            "name": name,
            "sector": sector,
            "price": round(latest, 1),
            "adjusted_price": adjusted_price,
            "1D": d1,
            "1W": w1,
            "1M": m1,
            "YTD": ytd,
            "1Y": y1
        })
    except Exception as e:
        print(f"スキップ ({code}): {e}")

# 3. 各銘柄のリアルな構成比（weight %）を算出
sectors = {}
for item in raw_items:
    sec = item["sector"]
    if sec not in sectors:
        sectors[sec] = []

    if total_adjusted_price > 0 and math.isfinite(total_adjusted_price):
        weight_pct = round((item["adjusted_price"] / total_adjusted_price) * 100, 2)
    else:
        weight_pct = 1.0

    # 面積が0になって消えるのを防ぐため最小値を0.05に設定
    if weight_pct <= 0:
        weight_pct = 0.05

    sectors[sec].append({
        "code": item["code"],
        "name": item["name"],
        "price": item["price"],
        "weight": weight_pct,
        "1D": item["1D"],
        "1W": item["1W"],
        "1M": item["1M"],
        "YTD": item["YTD"],
        "1Y": item["1Y"]
    })

output = {
    "updated_at": now_jst.strftime("%Y/%m/%d %H:%M"),
    "sectors": [
        {"name": sec, "children": items}
        for sec, items in sectors.items()
    ]
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

print(f"完了: {len(raw_items)}銘柄のデータを出力しました。")
