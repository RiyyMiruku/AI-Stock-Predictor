import joblib
import pandas as pd
import config

# 讀取最新特徵資料
df = pd.read_csv(f'data/{config.TICKER}_daily_features.csv', parse_dates=['date'])

# 準備預測資料（取最新一筆，不含 date, close, target）
X_pred = df.drop(['date', 'close', 'target'], axis=1).tail(1)

# 載入模型
model = joblib.load(config.MODEL_PATH)

# 預測
y_pred = model.predict(X_pred)
print("預測隔日漲跌幅：", y_pred[0])