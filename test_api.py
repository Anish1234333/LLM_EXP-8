"""Automated checks for validation, error handling and health.
These use a fake service, so they run without an API key or model download."""
import pytest
from fastapi.testclient import TestClient

import app as api
from support_rag import FALLBACK_ANSWER, BackendError


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def ask(self, question):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    api.app.dependency_overrides.clear()


def client_with(service):
    api.app.dependency_overrides[api.get_service] = lambda: service
    return TestClient(api.app)  # without "with", the lifespan (model loading) is skipped


def test_health_endpoint_reports_running():
    r = TestClient(api.app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_supported_question_is_answered():
    svc = FakeService(result={"answer": "Two working days.", "status": "answered",
                              "sources": ["bank_guide.txt"]})
    r = client_with(svc).post("/ask", json={"question": "How long does a cheque take to clear?"})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "answered"
    assert body["latency_ms"] >= 0


def test_unsupported_question_returns_fallback():
    svc = FakeService(result={"answer": FALLBACK_ANSWER,
                              "status": "insufficient_information", "sources": []})
    r = client_with(svc).post("/ask", json={"question": "What is the fixed deposit rate today?"})
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_information"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}])
def test_empty_question_is_rejected(payload):
    r = client_with(FakeService()).post("/ask", json=payload)
    assert r.status_code == 422


def test_question_over_1000_characters_is_rejected():
    r = client_with(FakeService()).post("/ask", json={"question": "a" * 1001})
    assert r.status_code == 422


def test_question_of_exactly_1000_characters_is_accepted():
    svc = FakeService(result={"answer": "ok", "status": "answered", "sources": ["bank_guide.txt"]})
    r = client_with(svc).post("/ask", json={"question": "a" * 1000})
    assert r.status_code == 200


def test_backend_failure_returns_controlled_503_without_secrets():
    svc = FakeService(error=BackendError("LLM call failed"))
    r = client_with(svc).post("/ask", json={"question": "How do I block my card?"})
    assert r.status_code == 503
    assert "hf_" not in r.text
    assert "Traceback" not in r.text
