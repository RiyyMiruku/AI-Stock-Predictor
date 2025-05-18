from fetcher import fetch_yahoo_finance_news
from deduplicator import deduplicate
from analyzer import analyze_with_groq
from daily_tech_factor import compute_technical_factors
from train_model import SimpleFactorModel
import pandas as pd
import json
import config
from datetime import datetime, timedelta
import os


if __name__ == '__main__':
    #抓取當日新聞
    news = fetch_yahoo_finance_news(config.NEWS_COUNT)

    print("[Step 1] 原始新聞：", len(news))

    #去除可能的相同新聞事件
    news = deduplicate(news)
    print("[Step 2] 去重後：", len(news))

    # 進行 LLM 分析，給出新聞對投資因子的評分
    rows = []
    for item in news:
        print("\n新聞標題：", item['title'])
        analysis = analyze_with_groq(item, config.TICKER)
        print("分析結果：", analysis)
        try:
            row = json.loads(analysis)
            # row['title'] = item['title']
            # row['content'] = item['content']
            rows.append(row)
        except json.JSONDecodeError:
            print("⚠️ 無法解析 JSON 格式")
    print("\n[Step 3] LLM 解析完畢：")

    #將各個新聞因子透過confidence作為權重合併成一筆"當日因子"

    df_news_factor = pd.DataFrame(rows)
    print(df_news_factor.head())

    numeric_features = ['sentiment_score', 'volatility_hint', 'confidence_level', 'aggregated_signal_score']
    weights = df_news_factor['confidence_level']
    weighted_avg = (df_news_factor[numeric_features].T * weights).T.sum() / weights.sum()
    daily_news_feature = weighted_avg.to_frame().T

    today = datetime.today().strftime("%Y-%m-%d")
    daily_news_feature.insert(0, 'date', today)  # 在第一列插入日期

    #抓取前幾日技術指標因子
    daily_tech_feature= compute_technical_factors(config.TICKER)
    print("\n[Step 4] 技術指標：")
    print(daily_tech_feature.tail(1))

    #合併新聞因子與技術指標因子
    tech = daily_tech_feature.tail(1).reset_index(drop=True)
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

    
    