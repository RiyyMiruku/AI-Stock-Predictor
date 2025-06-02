from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
import yfinance as yf
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import config

# 確保 model 資料夾存在
os.makedirs("model", exist_ok=True)

class SimpleFactorModel:
    def __init__(self):
        self.model = SGDRegressor()
        self.scaler = StandardScaler()

    def train(self, X, y):
        # 重新 fit scaler 並訓練模型
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 全量 fit
        self.model = SGDRegressor()
        self.model.fit(X_scaled, y)

        # 模型預測
        y_pred = self.model.predict(X_scaled)
        y_baseline = y.shift(1).fillna(0).values

        # 評估指標
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        base_mse = mean_squared_error(y, y_baseline)
        base_r2 = r2_score(y, y_baseline)

        print("模型已重新訓練（全資料）。")
        print(f"MSE: {mse:.6f}, R2: {r2:.4f}")
        print(f"Baseline MSE: {base_mse:.6f}, Baseline R2: {base_r2:.4f}")

        # 繪製 MSE 圖
        plt.figure()
        plt.bar(['Model MSE', 'Baseline MSE'], [mse, base_mse], color=['blue', 'gray'])
        plt.ylabel('MSE')
        plt.title('Model vs Baseline MSE')
        plt.tight_layout()
        plt.savefig('model/mse_comparison.png')
        plt.close()

        # 繪製 R2 圖
        plt.figure()
        plt.bar(['Model R2', 'Baseline R2'], [r2, base_r2], color=['green', 'gray'])
        plt.ylabel('R2 Score')
        plt.title('Model vs Baseline R2')
        plt.tight_layout()
        plt.savefig('model/r2_comparison.png')
        plt.close()

        # 預測 vs 實際
        plt.figure(figsize=(10, 4))
        plt.plot(y.values, label='Actual', color='black')
        plt.plot(y_pred, label='Predicted', color='blue', linestyle='--')
        plt.title('Predicted vs Actual Returns')
        plt.xlabel('Sample Index')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('model/pred_vs_actual.png')
        plt.close()

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def get_trained_until():
    if os.path.exists(config.TRAINED_UNTIL_PATH):
        with open(config.TRAINED_UNTIL_PATH, "r") as f:
            return pd.to_datetime(f.read().strip())
    return None

def update_trained_until(date):
    with open(config.TRAINED_UNTIL_PATH, "w") as f:
        f.write(str(date))

if __name__ == "__main__":
    csv_path = f'data/{config.TICKER}_daily_features.csv'
    df = pd.read_csv(csv_path, parse_dates=['date'])

    start_date = df['date'].min()
    end_date = df['date'].max() + pd.Timedelta(days=2)
    price = yf.download(f"{config.TICKER}", start=start_date, end=end_date)
    if isinstance(price.columns, pd.MultiIndex):
        price.columns = [str(col[0]) for col in price.columns]
    price = price.reset_index().rename(columns={'Date': 'date', 'Close': 'close'})
    price = price[['date', 'close']]

    if 'close' in df.columns:
        df = df.drop(columns=['close'])
    df = pd.merge(df, price, on='date', how='left')
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    df.to_csv(csv_path, index=False)
    print(f"已將隔日漲跌幅對齊並寫回 {csv_path}")

    try:
        df_train = pd.read_csv(csv_path, parse_dates=['date'])
        df_train = df_train.dropna(subset=['target'])

        X = df_train.drop(['date', 'close', 'target'], axis=1)
        y = df_train['target']

        model = SimpleFactorModel()
        model.train(X, y)
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(model.scaler, config.SCALER_PATH)
        update_trained_until(df_train['date'].max())

        print("模型與標準化器已訓練並保存。")

    except Exception as e:
        print(f"訓練過程發生錯誤：{e}")
