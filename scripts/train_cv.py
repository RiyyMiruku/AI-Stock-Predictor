"""TimeSeriesSplit 交叉驗證版的訓練腳本，輸出每折 MSE / R² 走勢圖。

執行方式：
    python -m scripts.train_cv
"""
import os
import joblib
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src import config
from src.model import SimpleFactorModel, update_trained_until


def cv_train(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    mse_list, base_mse_list = [], []
    r2_list, base_r2_list = [], []
    fold_idx = []

    model, scaler = None, None
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SGDRegressor()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

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

        print(
            f"Fold {i+1}: MSE={mse:.6f}, R2={r2:.4f} | "
            f"Baseline MSE={base_mse:.6f}, R2={base_r2:.4f}"
        )

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(fold_idx, mse_list, marker="o", label="Model MSE")
    plt.plot(fold_idx, base_mse_list, marker="x", linestyle="--", label="Baseline MSE")
    plt.xlabel("Fold")
    plt.ylabel("MSE")
    plt.title("MSE Comparison Across TimeSeries Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, "mse_timeseries_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(fold_idx, r2_list, marker="o", label="Model R2")
    plt.plot(fold_idx, base_r2_list, marker="x", linestyle="--", label="Baseline R2")
    plt.xlabel("Fold")
    plt.ylabel("R2 Score")
    plt.title("R2 Score Comparison Across TimeSeries Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, "r2_timeseries_plot.png"))
    plt.close()

    return model, scaler


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    csv_path = config.FEATURES_FILE
    df = pd.read_csv(csv_path, parse_dates=["date"])

    start_date = df["date"].min()
    end_date = df["date"].max() + pd.Timedelta(days=2)
    price = yf.download(config.TICKER, start=start_date, end=end_date)
    if isinstance(price.columns, pd.MultiIndex):
        price.columns = [str(col[0]) for col in price.columns]
    price = price.reset_index().rename(columns={"Date": "date", "Close": "close"})
    price = price[["date", "close"]]

    if "close" in df.columns:
        df = df.drop(columns=["close"])
    df = pd.merge(df, price, on="date", how="left")
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    df.to_csv(csv_path, index=False)
    print(f"已將隔日漲跌幅對齊並寫回 {csv_path}")

    try:
        df_train = pd.read_csv(csv_path, parse_dates=["date"])
        df_train = df_train.dropna(subset=["target"])

        X = df_train.drop(["date", "close", "target"], axis=1)
        y = df_train["target"]

        wrapper = SimpleFactorModel()
        last_model, last_scaler = cv_train(X, y)
        wrapper.model = last_model
        wrapper.scaler = last_scaler

        joblib.dump(wrapper, config.MODEL_PATH)
        joblib.dump(wrapper.scaler, config.SCALER_PATH)
        update_trained_until(df_train["date"].max())
        print("模型已訓練並完成交叉評估，已保存最後一折的模型。")
    except Exception as e:
        print(f"訓練過程發生錯誤：{e}")


if __name__ == "__main__":
    main()
