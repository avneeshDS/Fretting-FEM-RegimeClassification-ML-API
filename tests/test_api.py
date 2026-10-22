from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["Status"] == "Healthy"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "P": 50.0,
            "D": 0.025,
            "µ": 0.5,
            "a": 0.11804,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_regime" in result
    assert result["predicted_regime"] in [
        "Gross Slip",
        "Mixed Slip",
        "Partial Slip",
    ]

    assert "derived_features" in result