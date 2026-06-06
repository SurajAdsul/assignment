import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import build_retriever


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def retriever():
    return build_retriever()
