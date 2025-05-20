from sklearn.linear_model import SGDRegressor
import joblib
import pandas as pd
import yfinance as yf
import os
import yfinance as yf
import config
from sklearn.preprocessing import StandardScaler

class SimpleFactorModel:
    def __init__(self):
        self.model = SGDRegressor()
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X, y):
        if not self.is_trained:
            # 初次訓練，fit scaler
            X_scaled = self.scaler.fit_transform(X)
            self.model.partial_fit(X_scaled, y)
            self.is_trained = True
            print("模型初次已訓練完成。")
        else:
            # 後續增量訓練
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)
            print("完成一次增量訓練。")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
#==================================================

if __name__ == "__main__":
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

    # 合併收盤價到原df
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