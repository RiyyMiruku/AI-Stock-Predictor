from fetcher import fetch_yahoo_finance_news
from deduplicator import deduplicate
from analyzer import analyze_with_groq
from daily_tech_factor import compute_technical_factors
import pandas as pd
import json
import config
from datetime import datetime, timedelta
import os
import requests
import yfinance as yf


def is_today_market_closed(symbol: str) -> bool:
  
    today = datetime.now().date()
    data = yf.download(symbol, start=today, end=today + timedelta(days=1), progress=False)

    #因為yfinance如果在非交易日下載會回傳前一天的資料，所以需要檢查第一筆資料的日期是否為今天

    #若下載的筆數非空，判斷下載的資料的日期是否為今天
    if not data.index.empty:
        # 若第一筆資料的日期為今天，表示今天是交易日
        if data.index[0].date() == today :
            return True
    return False

if __name__ == '__main__':
     # 抓取當日新聞
    news = fetch_yahoo_finance_news(config.NEWS_COUNT)

    # 如果不是交易日，新聞先存到暫存檔，結束
    if not is_today_market_closed(config.TICKER):
        # 讀取舊的暫存新聞
        if os.path.exists(config.PENDING_NEWS_FILE):
            with open(config.PENDING_NEWS_FILE, "r", encoding="utf-8") as f:
                pending_news = json.load(f)
        else:
            pending_news = []
        # 合併新新聞
        pending_news.extend(news)
        # 去重
        pending_news = deduplicate(pending_news)
        # 存回暫存檔
        with open(config.PENDING_NEWS_FILE, "w", encoding="utf-8") as f:
            json.dump(pending_news, f, ensure_ascii=False, indent=2)
        print("新聞已暫存，等待下次交易日處理。")
        exit()

    # 如果是交易日，讀取暫存新聞+當日新聞
    if os.path.exists(config.PENDING_NEWS_FILE):
        with open(config.PENDING_NEWS_FILE, "r", encoding="utf-8") as f:
            pending_news = json.load(f)
        news.extend(pending_news)
        os.remove(config.PENDING_NEWS_FILE)  # 清空暫存
        print(f"合併處理暫存新聞，共 {len(news)} 則")

    # 去重
    news = deduplicate(news)
    print("[Step 2] 去重後：", len(news))

    # 進行 LLM 分析，給出新聞對投資因子的評分
    rows = []
    for item in news:
        print("\n新聞標題：", item['title'])
        analysis = analyze_with_groq(item, config.TICKER)
        print("分析結果：", analysis)
        rows.append(analysis)
    print("\n[Step 3] LLM 解析完畢：")

    #將各個新聞因子透過confidence作為權重合併成一筆"當日因子"

    df_news_factor = pd.DataFrame(rows)
    print(df_news_factor.head())

    numeric_features = ['sentiment_score', 'volatility_hint', 'confidence_level', 'aggregated_signal_score', 'positive_neutral_negative']
    weights = df_news_factor['confidence_level']
    weighted_avg = (df_news_factor[numeric_features].T * weights).T.sum() / weights.sum()
    daily_news_feature = weighted_avg.to_frame().T

    today = datetime.today().strftime("%Y-%m-%d")
    daily_news_feature.insert(0, 'date', today)  # 在第一列插入日期

    #抓取前幾日技術指標因子
    daily_tech_feature= compute_technical_factors(config.TICKER)
    print("\n[Step 4] 技術指標：")
    print(daily_tech_feature)

    #合併新聞因子與技術指標因子
    tech = daily_tech_feature.reset_index(drop=True)
    tech.columns = [f"tech_{col}" for col in tech.columns]  # 強制單層欄位名
    daily_feature = pd.concat([daily_news_feature, tech], axis=1)

    # 儲存當日新聞、技術指標因子
    os.makedirs("data", exist_ok=True)
    features_file = f"data/{config.TICKER}_daily_features.csv"

    if os.path.exists(features_file):
        date_col = 'date'
        df_all = pd.read_csv(features_file)
        date_to_update = daily_feature[date_col].iloc[0]
        df_all = df_all[df_all[date_col] != date_to_update]
        daily_feature = daily_feature.reindex(columns=df_all.columns)  # 欄位順序對齊
        df_all = pd.concat([df_all, daily_feature], ignore_index=True)
    else:
        df_all = daily_feature
    
    # 展平成單層欄位
    df_all.columns = [str(col) for col in df_all.columns]
    df_all.to_csv(features_file, index=False)

    
    