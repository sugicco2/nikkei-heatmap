import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta

# 1. 銘柄リストを読み込む
df_tickers = pd.read_csv("tickers.csv").drop_duplicates(subset=["code"])
codes = df_tickers["code"].tolist()

# 2. 過去2年分の株価を一括ダウンロード
print("株価データを取得中...")
data = yf.download(codes, period="2y", group_by="ticker", auto_adjust=True, threads=True)

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)
current_year = now_jst.year

raw_items = []
total_adjusted_price = 0.0

for _, row in df_tickers.iterrows():
    code = row["code"]
    name = row["name"]
    sector = row["sector"]
    factor = float(row["factor"])

    try:
        if len(codes) == 1:
            closes = data["Close"].dropna()
        else:
            closes = data[code]["Close"].dropna()

        if len(closes) < 2:
            continue

        latest = float(closes.iloc[-1])

        # 日経平均の換算株価（株価 × 株価換算係数）を計算して合計に足す
        adjusted_price = latest * factor
        total_adjusted_price += adjusted_price

        def calc_change(idx_offset):
            if len(closes) > idx_offset:
                prev = float(closes.iloc[-(idx_offset + 1)])
                return round(((latest - prev) / prev) * 100, 2)
            return 0.0

        d1 = calc_change(1)
        w1 = calc_change(5)
        m1 = calc_change(21)
        y1 = calc_change(245)

        prev_year_closes = closes[closes.index.year < current_year]
        if len(prev_year_closes) > 0:
            ytd_base = float(prev_year_closes.iloc[-1])
            ytd = round(((latest - ytd_base) / ytd_base) * 100, 2)
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

# 3. 全体の換算株価合計から、各銘柄のリアルな構成比（weight %）を算出
sectors = {}
for item in raw_items:
    sec = item["sector"]
    if sec not in sectors:
        sectors[sec] = []

    weight_pct = round((item["adjusted_price"] / total_adjusted_price) * 100, 2) if total_adjusted_price > 0 else 0.0

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
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

print("最新の構成比（ウェイト）反映済みの data.json を作成しました！")
