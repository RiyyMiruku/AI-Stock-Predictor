from sklearn.linear_model import SGDRegressor
import joblib
import pandas as pd
import yfinance as yf
import os
import yfinance as yf
import config

class SimpleFactorModel:
    def __init__(self):
        self.model = SGDRegressor()
        self.is_trained = False

    def train(self, X, y):
        if not self.is_trained:
            # 初次訓練
            self.model.partial_fit(X, y)
            self.is_trained = True
        else:
            # 增量訓練
            self.model.partial_fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
#==================================================

# # 讀取原有特徵資料
csv_path = f'data/{config.TICKER}_daily_features.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])

# 取得日期範圍
start_date = df['date'].min()
end_date = df['date'].max() + pd.Timedelta(days=2)  # 多抓一天以便計算隔日漲幅

# 下載收盤價
price = yf.download(f"{config.TICKER}", start=start_date, end=end_date)
if isinstance(price.columns, pd.MultiIndex):
    price.columns = [str(col[0]) for col in price.columns]
price = price.reset_index().rename(columns={'Date': 'date', 'Close': 'close'})
price = price[['date', 'close']]

# 如果原本有 close 欄位，先刪除
if 'close' in df.columns:
    df = df.drop(columns=['close'])

# 合併收盤價到原df，這時只會有一個 close 欄位
df = pd.merge(df, price, on='date', how='left')

# 用合併後的 close 欄位計算 target
df['target'] = df['close'].shift(-1) / df['close'] - 1


# 覆蓋寫回原有 CSV
df.to_csv(csv_path, index=False)
print(f"已將隔日漲跌幅對齊並寫回 {csv_path}")

#==================================================

try:
    # 讀取新 CSV 進行訓練
    df_train = pd.read_csv(csv_path)

    # 僅保留 target 不為 NaN 的資料
    df_train = df_train.dropna(subset=['target'])

    # 準備訓練資料
    X = df_train.drop(['date', 'close', 'target'], axis=1)
    y = df_train['target']

    # 載入或建立模型
    if os.path.exists(config.MODEL_PATH):
        model = joblib.load(config.MODEL_PATH)
    else:
        model = SimpleFactorModel()

    # 訓練模型
    model.train(X, y)

    # 保存模型
    joblib.dump(model, config.MODEL_PATH)

    print("模型已訓練並保存。")

except Exception as e:
    print(f"訓練過程發生錯誤：{e}")