"""Tests for the main application."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vector_db.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_api_docs_available():
    """Test that API documentation is available."""
    response = client.get("/api/v1/docs")
    assert response.status_code == status.HTTP_200_OK
