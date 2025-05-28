import pandas as pd
import joblib
import config
from train_model import SimpleFactorModel

import pandas as pd
import joblib
import config
from train_model import SimpleFactorModel
import os
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "model/scaler.save"

def predict_next_return():
    # 讀取最新特徵資料
    df = pd.read_csv(f"data/{config.TICKER}_daily_features.csv", parse_dates=['date'])
    X_pred = df.drop(['date', 'close', 'target'], axis=1).tail(1)

    # 載入模型與 scaler
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("模型或標準化器檔案不存在，請先訓練模型。")

    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 標準化並預測
    X_scaled = scaler.transform(X_pred)
    y_pred = model.model.predict(X_scaled)
    print(f"預測隔日漲跌幅：{y_pred[0]}")
    return y_pred[0]

# 範例呼叫
if __name__ == "__main__":
    predict_next_return()