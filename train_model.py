from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = Path("models/model.joblib")
CATEGORIES = ["grocery", "electronics", "travel", "entertainment", "other"]


def build_training_data(n: int = 2500, seed: int = 42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user_id": rng.integers(1, 5000, n),
        "amount": rng.uniform(10, 100000, n),
        "merchant_category": rng.choice(CATEGORIES, n),
        "location_lat": rng.uniform(-90, 90, n),
        "location_lon": rng.uniform(-180, 180, n),
        "is_international": rng.choice([False, True], n, p=[0.82, 0.18]),
    })
    score = (
        0.000035 * df["amount"]
        + 1.4 * df["is_international"].astype(int)
        + 0.7 * df["merchant_category"].isin(["electronics", "travel"]).astype(int)
        + rng.normal(0, 0.8, n)
    )
    y = (score > 2.8).astype(int)
    return df, y


def train_and_save():
    X, y = build_training_data()
    numeric = ["user_id", "amount", "location_lat", "location_lon"]
    categorical = ["merchant_category", "is_international"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
