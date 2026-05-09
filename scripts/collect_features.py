"""每日資料收集腳本：抓新聞 → 去重 → LLM 評分 → 合併技術指標 → 寫入 CSV。

非交易日抓到的新聞會暫存到 PENDING_NEWS_FILE，等到下個交易日一併處理。

執行方式：
    python -m scripts.collect_features
"""
import os
import json
from datetime import datetime

import pandas as pd

from src import config
from src.fetcher import fetch_yahoo_finance_news
from src.deduplicator import deduplicate
from src.analyzer import analyze_with_groq
from src.tech_factors import compute_technical_factors
from src.market_calendar import is_today_market_open


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 抓取當日新聞
    news = fetch_yahoo_finance_news(config.NEWS_COUNT)

    # 非交易日：累積到暫存檔，結束
    if not is_today_market_open(config.TICKER):
        if os.path.exists(config.PENDING_NEWS_FILE):
            with open(config.PENDING_NEWS_FILE, "r", encoding="utf-8") as f:
                pending_news = json.load(f)
        else:
            pending_news = []
        pending_news.extend(news)
        pending_news = deduplicate(pending_news)
        with open(config.PENDING_NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(pending_news, f, ensure_ascii=False, indent=2)
        print("新聞已暫存，等待下次交易日處理。")
        return

    # 交易日：合併暫存新聞
    if os.path.exists(config.PENDING_NEWS_FILE):
        with open(config.PENDING_NEWS_FILE, "r", encoding="utf-8") as f:
            pending_news = json.load(f)
        news.extend(pending_news)
        os.remove(config.PENDING_NEWS_FILE)
        print(f"合併處理暫存新聞，共 {len(news)} 則")

    news = deduplicate(news)
    print("[Step 2] 去重後：", len(news))

    # LLM 分析
    rows = []
    for item in news:
        print("\n新聞標題：", item["title"])
        analysis = analyze_with_groq(item, config.TICKER)
        print("分析結果：", analysis)
        rows.append(analysis)
    print("\n[Step 3] LLM 解析完畢：")

    # 以 confidence 為權重合併出當日新聞因子
    df_news_factor = pd.DataFrame(rows)
    print(df_news_factor.head())

    numeric_features = [
        "sentiment_score",
        "volatility_hint",
        "confidence_level",
        "positive_neutral_negative",
    ]
    weights = df_news_factor["confidence_level"]
    weighted_avg = (df_news_factor[numeric_features].T * weights).T.sum() / weights.sum()
    daily_news_feature = weighted_avg.to_frame().T

    today = datetime.today().strftime("%Y-%m-%d")
    daily_news_feature.insert(0, "date", today)

    # 技術指標
    daily_tech_feature = compute_technical_factors(config.TICKER)
    print("\n[Step 4] 技術指標：")
    print(daily_tech_feature)

    # 合併新聞因子與技術指標因子
    tech = daily_tech_feature.reset_index(drop=True)
    tech.columns = [f"tech_{col}" for col in tech.columns]
    daily_feature = pd.concat([daily_news_feature, tech], axis=1)

    # 寫入 / 更新 CSV
    if os.path.exists(config.FEATURES_FILE):
        df_all = pd.read_csv(config.FEATURES_FILE)
        date_to_update = daily_feature["date"].iloc[0]
        df_all = df_all[df_all["date"] != date_to_update]
        daily_feature = daily_feature.reindex(columns=df_all.columns)
        df_all = pd.concat([df_all, daily_feature], ignore_index=True)
    else:
        df_all = daily_feature

    df_all.columns = [str(col) for col in df_all.columns]
    df_all.to_csv(config.FEATURES_FILE, index=False)


if __name__ == "__main__":
    main()
