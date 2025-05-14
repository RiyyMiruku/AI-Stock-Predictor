from fetcher import fetch_news
from deduplicator import deduplicate
from analyzer import analyze_article
from factors import compute_technical_factors
from model import SimpleFactorModel
import pandas as pd
import json

if __name__ == '__main__':
    news = fetch_news()
    print("[Step 1] 原始新聞：", len(news))

    news = deduplicate(news)
    print("[Step 2] 去重後：", len(news))

    rows = []
    for item in news:
        print("\n新聞標題：", item['title'])
        analysis = analyze_article(item['title'])
        print("分析結果：", analysis)
        try:
            row = json.loads(analysis)
            rows.append(row)
        except:
            print("⚠️ 無法解析 JSON 格式")

    df_news = pd.DataFrame(rows)
    print("\n[Step 3] LLM 解析完畢：")
    print(df_news.head())

    df_tech = compute_technical_factors("^TWII")
    print("\n[Step 4] 技術指標：")
    print(df_tech.tail(1))

    # 模擬合併因子（注意：這裡簡化了資料對齊問題）
    if not df_news.empty:
        df_news.fillna(0, inplace=True)
        dummy_X = pd.concat([df_news.select_dtypes("number"), df_tech.tail(1).reset_index(drop=True)], axis=1)
        dummy_y = [0.01]  # 假設隔日報酬

        model = SimpleFactorModel()
        model.train(dummy_X, dummy_y)
        prediction = model.predict(dummy_X)

        print("\n[Step 5] 模型預測結果：E(r) =", prediction[0])

    print("\n[✔] 分析流程完成。")
