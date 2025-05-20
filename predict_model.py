import pandas as pd
import joblib
import config
from train_model import SimpleFactorModel

def predict_next_return():
    # 讀取最新特徵資料
    df = pd.read_csv(f"data/{config.TICKER}_daily_features.csv", parse_dates=['date'])
    # 取最新一筆，去除無關欄位
    X_pred = df.drop(['date', 'close', 'target'], axis=1).tail(1)
    # 載入模型
    model = joblib.load(config.MODEL_PATH)
    # 預測
    y_pred = model.predict(X_pred)
    print(f"預測隔日漲跌幅：{y_pred[0]}")
    return y_pred[0]

# 範例呼叫
if __name__ == "__main__":
    predict_next_return()