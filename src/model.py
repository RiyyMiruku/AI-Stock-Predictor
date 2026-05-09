import os
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

from src import config


class SimpleFactorModel:
    """以 SGDRegressor + StandardScaler 預測隔日漲跌幅的線性模型。"""

    def __init__(self):
        self.model = SGDRegressor()
        self.scaler = StandardScaler()

    def train(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = SGDRegressor()
        self.model.fit(X_scaled, y)

        y_pred = self.model.predict(X_scaled)
        y_baseline = y.shift(1).fillna(0).values

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        base_mse = mean_squared_error(y, y_baseline)
        base_r2 = r2_score(y, y_baseline)

        print("模型已重新訓練（全資料）。")
        print(f"MSE: {mse:.6f}, R2: {r2:.4f}")
        print(f"Baseline MSE: {base_mse:.6f}, Baseline R2: {base_r2:.4f}")

        os.makedirs(config.MODEL_DIR, exist_ok=True)

        plt.figure()
        plt.bar(["Model MSE", "Baseline MSE"], [mse, base_mse], color=["blue", "gray"])
        plt.ylabel("MSE")
        plt.title("Model vs Baseline MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_DIR, "mse_comparison.png"))
        plt.close()

        plt.figure()
        plt.bar(["Model R2", "Baseline R2"], [r2, base_r2], color=["green", "gray"])
        plt.ylabel("R2 Score")
        plt.title("Model vs Baseline R2")
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_DIR, "r2_comparison.png"))
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(y.values, label="Actual", color="black")
        plt.plot(y_pred, label="Predicted", color="blue", linestyle="--")
        plt.title("Predicted vs Actual Returns")
        plt.xlabel("Sample Index")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_DIR, "pred_vs_actual.png"))
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
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.TRAINED_UNTIL_PATH, "w") as f:
        f.write(str(date))
