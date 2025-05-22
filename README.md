# AI_Stock_Predictor

AI_Stock_Predictor 是一個自動化台股指數預測系統，結合每日新聞情緒分析與技術指標，持續累積資料並訓練機器學習模型，預測隔日台股指數漲跌幅。

## 特色

- 自動抓取 Yahoo 財經新聞
- LLM 分析新聞情緒與投資因子
- 計算台股技術指標（動能、量比、波動率等）
- 每日自動累積特徵資料到 CSV
- 支援線性模型訓練與保存
- 可隨時進行隔日漲跌幅預測

## 安裝需求

- Python 3.8+
- 主要套件：`pandas`, `yfinance`, `requests`, `beautifulsoup4`, `scikit-learn`, `joblib`
- 需設定 `.env` 或 `config.py` 以提供 API 金鑰與參數

## 使用說明

### 1. 安裝套件

```bash
pip install -r requirements.txt
or
uv sync
```

### 2. 每日自動執行資料收集與特徵累積

```bash
python news_to_index.py
```

### 3. 訓練模型

```bash
python train_model.py
```

### 4. 預測隔日漲跌幅

```bash
python predict_model.py
```

## 專案結構

```
AI_Stock_Predictor/
│
├── analyzer.py           # LLM 分析新聞
├── fetcher.py            # 抓取 Yahoo 財經新聞
├── deduplicator.py       # 新聞去重復
├── daily_tech_factor.py  # 計算技術指標
├── news_to_index.py      # 主流程，合併特徵並累積資料
├── train_model.py        # 訓練模型
├── predict_model.py      # 預測模型
├── data/                 # 儲存每日特徵與模型
│   └── ^TWII_daily_features.csv
│   └── simple_factor_model.joblib
├── config.py             # 參數設定
└── README.md
```

## 注意事項

- 請搭配排程工具（如 crontab 或 Windows 工作排程器）每日自動執行 `news_to_index.py`。
- 若要更換預測標的，請修改 `config.py` 內的 TICKER 參數。
- 本專案僅供學術與研究用途，預測結果不構成任何投資建議。

## License

MIT# AI_Stock_Predictor
collect daily news to give a predictive trend of stock market 
