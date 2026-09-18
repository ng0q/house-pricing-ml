import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import model_loader
from app.main import app

VALID_PAYLOAD = {
    "location": "thane",
    "transaction": 1,
    "furnishing": 1,
    "facing": "east",
    "bathroom": 2,
    "balcony": 1,
    "car_covered": 1,
    "parking_count": 1,
    "super_area_sqft": 1000,
    "carpet_area_sqft": 800,
    "floor_current": 10,
    "floor_total": 16,
}


class DummyEncoder:
    def transform(self, df):
        return df


class DummyModel:
    def predict(self, X):
        return np.array([np.log1p(7_955_000)])


@pytest.fixture
def client(monkeypatch):
    def fake_load():
        model_loader.model = DummyModel()
        model_loader.target_encoder = DummyEncoder()

    monkeypatch.setattr(model_loader, "load_artifacts", fake_load)
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_response_structure(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predicted_price", "predicted_price_formatted"}
    assert isinstance(body["predicted_price"], float)
    assert isinstance(body["predicted_price_formatted"], str)


@pytest.mark.parametrize("transaction", [-1, 2])
def test_transaction_validation(client, transaction):
    payload = {**VALID_PAYLOAD, "transaction": transaction}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
