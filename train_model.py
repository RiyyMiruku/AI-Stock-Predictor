"""向後相容用：舊版 joblib pickle 會引用 `train_model.SimpleFactorModel`。

新程式請改用 `from src.model import SimpleFactorModel`。
"""
from src.model import SimpleFactorModel, get_trained_until, update_trained_until  # noqa: F401
