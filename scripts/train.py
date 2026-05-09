"""對齊隔日漲跌幅標籤後，以全資料訓練 SimpleFactorModel。

執行方式：
    python -m scripts.train
"""
import os
import joblib
import pandas as pd
import yfinance as yf

from src import config
from src.model import SimpleFactorModel, update_trained_until


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    csv_path = config.FEATURES_FILE
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # 補入收盤價並計算隔日漲跌幅
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

        model = SimpleFactorModel()
        model.train(X, y)
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(model.scaler, config.SCALER_PATH)
        update_trained_until(df_train["date"].max())

        print("模型與標準化器已訓練並保存。")
    except Exception as e:
        print(f"訓練過程發生錯誤：{e}")


if __name__ == "__main__":
    main()
