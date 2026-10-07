"""
Unit Tests for FastAPI Service
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import QueryRequest, QueryResponse


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models" in data
    assert "total_vectors" in data


def test_valid_query_endpoint(client):
    payload = {"question": "What is the retention period for audit logs?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["answer"]) > 0


def test_empty_question_rejected(client):
    payload = {"question": ""}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_whitespace_question_rejected(client):
    payload = {"question": "   \n\t  "}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_excessive_length_question_rejected(client):
    payload = {"question": "A" * 2005}
    response = client.post("/query", json=payload)
    assert response.status_code == 422
