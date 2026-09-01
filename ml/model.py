from pathlib import Path
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

MODEL_PATH = Path(__file__).resolve().parent / "expense_forecaster.joblib"


class ExpenseForecaster:
    def __init__(self, db_path):
        self.db_path = db_path
        self.model = None
        self.metrics = {}
        self.train()

    def _load_daily(self):
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT date, amount FROM expenses", conn)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date", as_index=False)["amount"].sum().sort_values("date")

    def train(self):
        df = self._load_daily()
        if len(df) < 3:
            self.model = None
            return None
        df["day_num"] = (df["date"] - df["date"].min()).dt.days
        X = df[["day_num"]].values
        y = df["amount"].values
        split = max(2, int(len(df) * 0.8))
        if split >= len(df):
            split = len(df) - 1
        self.model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                               ("reg", LinearRegression())])
        self.model.fit(X[:split], y[:split])
        pred = np.maximum(0, self.model.predict(X[split:]))
        actual = y[split:]
        self.metrics = {
            "mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred)) if len(actual) > 1 else None,
            "training_days": int(len(df))
        }
        # Retrain final model on all available data for production predictions.
        self.model.fit(X, y)
        joblib.dump({"model": self.model, "metrics": self.metrics}, MODEL_PATH)
        return self.metrics

    def predict(self, days=7):
        df = self._load_daily()
        if len(df) < 3:
            return {"ready": False, "message": "Add expenses across at least 3 different days before using ML forecasts.", "predictions": [], "metrics": {}}
        if self.model is None:
            self.train()
        last_date = df["date"].max()
        start_num = int((last_date - df["date"].min()).days)
        future_nums = np.arange(start_num + 1, start_num + days + 1).reshape(-1, 1)
        amounts = np.maximum(0, self.model.predict(future_nums))
        predictions = [{"date": (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                        "amount": round(float(a), 2)} for i, a in enumerate(amounts)]
        return {"ready": True, "predictions": predictions, "metrics": self.metrics}
