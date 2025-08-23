"""
API endpoint testleri - Coverage artırmak için.
Pipeline uyumlu model testleri.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from addrnorm.api.server import app


@pytest.fixture(scope="module")
def client():
    """Test client fixture that properly handles startup/shutdown."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data


def test_normalize_single_success(client):
    """Tekli adres normalization testi."""
    response = client.post(
        "/normalize", json={"text": "İstanbul Beşiktaş Levent Mahallesi"}
    )
    assert response.status_code == 200
    data = response.json()

    # ProcessingResult yapısını kontrol et
    assert "raw_input" in data
    assert "success" in data
    assert "address_out" in data
    assert "processing_method" in data
    assert "confidence" in data
    assert "processing_time_ms" in data

    assert data["raw_input"] == "İstanbul Beşiktaş Levent Mahallesi"
    assert isinstance(data["success"], bool)
    assert isinstance(data["confidence"], (int, float)) or data["confidence"] is None
    assert (
        isinstance(data["processing_time_ms"], (int, float))
        or data["processing_time_ms"] is None
    )


def test_normalize_single_empty(client):
    """Boş input testi."""
    response = client.post("/normalize", json={"text": ""})
    # Validation hatası bekleniyor
    assert response.status_code == 422


def test_normalize_single_invalid_json(client):
    """Geçersiz JSON testi."""
    response = client.post("/normalize", json={"wrong_field": "test"})
    assert response.status_code == 422


def test_normalize_batch_success(client):
    """Batch normalization testi."""
    response = client.post(
        "/normalize/batch",
        json={"texts": ["İstanbul Beşiktaş", "Ankara Çankaya", "İzmir Konak"]},
    )
    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert "total_count" in data
    assert "success_count" in data
    assert "error_count" in data

    assert len(data["results"]) == 3
    assert data["total_count"] == 3

    # Her result ProcessingResult yapısında olmalı
    for result in data["results"]:
        assert "raw_input" in result
        assert "success" in result
        assert "address_out" in result


def test_normalize_batch_empty(client):
    """Boş batch testi."""
    response = client.post("/normalize/batch", json={"texts": []})
    assert response.status_code == 422


def test_normalize_batch_too_many(client):
    """Çok fazla adres testi (limit aşımı)."""
    texts = [f"Test adres {i}" for i in range(101)]
    response = client.post("/normalize/batch", json={"texts": texts})
    assert response.status_code == 422


def test_api_response_structure(client):
    """API response yapısının pipeline çıktısıyla uyumluluğu."""
    response = client.post("/normalize", json={"text": "Ankara Kızılay"})
    assert response.status_code == 200
    data = response.json()

    # ProcessingResult field'ları
    expected_fields = {
        "raw_input",
        "success",
        "address_out",
        "error",
        "processing_method",
        "confidence",
        "processing_time_ms",
    }
    assert set(data.keys()) == expected_fields


def test_normalize_turkish_characters(client):
    """Türkçe karakter testleri."""
    response = client.post("/normalize", json={"text": "İstanbul Üsküdar Çamlıca"})
    assert response.status_code == 200
    data = response.json()
    assert data["raw_input"] == "İstanbul Üsküdar Çamlıca"
    assert isinstance(data["success"], bool)
