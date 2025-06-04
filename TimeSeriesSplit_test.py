from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
        tscv = TimeSeriesSplit(n_splits=5)
        mse_list, base_mse_list = [], []
        r2_list, base_r2_list = [], []
        fold_idx = []

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SGDRegressor()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # baseline 是用 y_test 的前一天做預測
            y_baseline = y_test.shift(1).fillna(0).values

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            base_mse = mean_squared_error(y_test, y_baseline)
            base_r2 = r2_score(y_test, y_baseline)

            mse_list.append(mse)
            base_mse_list.append(base_mse)
            r2_list.append(r2)
            base_r2_list.append(base_r2)
            fold_idx.append(i + 1)

            print(f"Fold {i+1}: MSE={mse:.6f}, R2={r2:.4f} | Baseline MSE={base_mse:.6f}, R2={base_r2:.4f}")

        # 繪圖
        plt.figure(figsize=(8, 4))
        plt.plot(fold_idx, mse_list, marker='o', label='Model MSE')
        plt.plot(fold_idx, base_mse_list, marker='x', linestyle='--', label='Baseline MSE')
        plt.xlabel('Fold')
        plt.ylabel('MSE')
        plt.title('MSE Comparison Across TimeSeries Folds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('model/mse_timeseries_plot.png')
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(fold_idx, r2_list, marker='o', label='Model R2')
        plt.plot(fold_idx, base_r2_list, marker='x', linestyle='--', label='Baseline R2')
        plt.xlabel('Fold')
        plt.ylabel('R2 Score')
        plt.title('R2 Score Comparison Across TimeSeries Folds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('model/r2_timeseries_plot.png')
        plt.close()

        # 最後一次 fold 的模型與 scaler 作為最終模型保存
        self.model = model
        self.scaler = scaler
        print("模型已訓練並完成交叉評估。")

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
