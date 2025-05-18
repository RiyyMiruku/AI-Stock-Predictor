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

# 讀取特徵資料
df = pd.read_csv(f'data/{config.TICKER}_daily_features.csv', parse_dates=['date'])

# 下載指數收盤價
price = yf.download(f"{config.TICKER}", start=df['date'].min(), end=df['date'].max() + pd.Timedelta(days=2))
price = price['Close'].reset_index().rename(columns={'Date': 'date', 'Close': 'close'})

# 合併
df = pd.merge(df, price, on='date', how='left')

# 計算隔日漲跌幅
df['target'] = df['close'].shift(-1) / df['close'] - 1

# 移除最後一天（因為沒有隔日收盤價）
df = df.dropna(subset=['target'])

#==================================================

# 準備訓練資料
X = df.drop(['date', 'close', 'target'], axis=1)
y = df['target']

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