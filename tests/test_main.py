from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from train_model import train_and_save

MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    train_and_save()

from src.main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def valid_transaction():
    return {
        "user_id": 42,
        "amount": 1500.50,
        "merchant_category": "grocery",
        "location_lat": 50.45,
        "location_lon": 30.52,
        "is_international": False,
    }


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid(client, valid_transaction):
    response = client.post("/v1/predict", json=valid_transaction)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["is_fraud"] == (body["fraud_probability"] >= 0.5)
    assert "transaction_id" in body
    assert "timestamp" in body


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", 0),
        ("amount", -1),
        ("amount", 100001),
        ("merchant_category", "unknown"),
        ("location_lat", 91),
        ("location_lon", 181),
    ],
)
def test_validation_rejects_invalid_input(client, valid_transaction, field, value):
    payload = valid_transaction.copy()
    payload[field] = value
    assert client.post("/v1/predict", json=payload).status_code == 422


def test_batch_prediction(client, valid_transaction):
    response = client.post("/v1/predict/batch", json=[valid_transaction, valid_transaction])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_batch_limit(client, valid_transaction):
    response = client.post("/v1/predict/batch", json=[valid_transaction] * 51)
    assert response.status_code == 400
