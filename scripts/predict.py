"""讀取最新一筆特徵並預測隔日漲跌幅。

執行方式：
    python -m scripts.predict
"""
import os
import joblib
import pandas as pd

from src import config
from src.model import SimpleFactorModel  # noqa: F401  (joblib 反序列化需要)


def predict_next_return() -> float:
    df = pd.read_csv(config.FEATURES_FILE, parse_dates=["date"])
    X_pred = df.drop(["date", "close", "target"], axis=1).tail(1)

    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError("模型或標準化器檔案不存在，請先訓練模型。")

    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    X_scaled = scaler.transform(X_pred)
    y_pred = model.model.predict(X_scaled)
    print(f"預測隔日漲跌幅：{y_pred[0]}")
    return y_pred[0]


if __name__ == "__main__":
    predict_next_return()
