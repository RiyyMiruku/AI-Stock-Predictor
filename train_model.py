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
        self.is_trained = False

    def train(self, X, y):
        mse_list = []
        r2_list = []
        baseline_mse_list = []
        baseline_r2_list = []
        sample_counts = []

        if not self.is_trained:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            for i in range(len(X_scaled)):
                Xi = X_scaled[i].reshape(1, -1)
                yi = np.array([y[i]])
                self.model.partial_fit(Xi, yi)

                y_true = y[:i+1]
                y_pred = self.model.predict(X_scaled[:i+1])

                if i >= 1:
                    y_baseline = np.roll(y_true, 1)
                    y_baseline[0] = 0
                else:
                    y_baseline = np.zeros_like(y_true)

                mse = mean_squared_error(y_true, y_pred)
                base_mse = mean_squared_error(y_true, y_baseline)

                if len(y_true) >= 2:
                    r2 = r2_score(y_true, y_pred)
                    base_r2 = r2_score(y_true, y_baseline)
                else:
                    r2 = np.nan
                    base_r2 = np.nan

                mse_list.append(mse)
                r2_list.append(r2)
                baseline_mse_list.append(base_mse)
                baseline_r2_list.append(base_r2)
                sample_counts.append(i + 1)
            self.is_trained = True
            print("模型初次訓練（逐筆）完成。")

            if not os.path.exists(config.FIRST_TRAIN_FLAG):
                df_save = X.copy()
                df_save['target'] = y.values if isinstance(y, pd.Series) else y
                df_save.to_csv("model/first_train_data.csv", index=False)
                with open(config.FIRST_TRAIN_FLAG, "w") as f:
                    f.write("done")
                print("已儲存第一次訓練資料為 model/first_train_data.csv")
        else:
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)
            y_pred = self.model.predict(X_scaled)

            y_baseline = y.shift(1).fillna(0).values

            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            base_mse = mean_squared_error(y, y_baseline)
            base_r2 = r2_score(y, y_baseline)

            mse_list.append(mse)
            r2_list.append(r2)
            baseline_mse_list.append(base_mse)
            baseline_r2_list.append(base_r2)
            sample_counts.append(len(y))
            print("完成一次增量訓練。")

        # 繪圖
        if mse_list and r2_list:
            plt.figure()
            plt.plot(sample_counts, mse_list, label='Model MSE')
            plt.plot(sample_counts, baseline_mse_list, label='Baseline MSE', linestyle='--')
            plt.plot(sample_counts, r2_list, label='Model R2')
            plt.plot(sample_counts, baseline_r2_list, label='Baseline R2', linestyle='--')
            plt.xlabel('Training Samples')
            plt.ylabel('Score')
            plt.title('Training MSE and R2 vs Baseline')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('model/training_metrics.png')
            plt.close()
            print("已儲存訓練過程評估圖表 model/training_metrics.png")

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

        trained_until = get_trained_until()
        if trained_until:
            df_train = df_train[df_train['date'] > trained_until]

        if df_train.empty:
            print("沒有新資料可訓練，跳過。")
        else:
            X = df_train.drop(['date', 'close', 'target'], axis=1)
            y = df_train['target']

            if os.path.exists(config.MODEL_PATH):
                model = joblib.load(config.MODEL_PATH)
            else:
                model = SimpleFactorModel()

            model.train(X, y)
            joblib.dump(model, config.MODEL_PATH)
            joblib.dump(model.scaler, config.SCALER_PATH)
            update_trained_until(df_train['date'].max())

            print("模型與標準化器已訓練並保存。")

    except Exception as e:
        print(f"訓練過程發生錯誤：{e}")
