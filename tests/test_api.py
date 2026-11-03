"""Integration tests for serving/api.py.

Uses FastAPI's TestClient (no real server/network needed) and only
exercises endpoints that work without external API keys — the
SymPy-only solve/explain/chat/practice/profile paths. This keeps CI
fast and free of flaky network dependencies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serving.api import app


@pytest.fixture()
def client(tmp_path) -> TestClient:
    # Redirect student-profile storage to a temp dir so tests never
    # pollute (or depend on) real on-disk profile data.
    import serving.api as api_module
    from src.student_profile import StudentProfileStore

    api_module.service.profile_store = StudentProfileStore(storage_dir=tmp_path)
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["sympy_solver"] is True


class TestSolveLatex:
    def test_quadratic_solves_correctly(self, client: TestClient) -> None:
        response = client.post("/solve/latex", json={"latex": "x^2 - 5x + 6 = 0"})
        assert response.status_code == 200
        body = response.json()
        assert body["problem_type"] == "quadratic_equation"
        assert "2" in body["answer"] and "3" in body["answer"]
        assert len(body["steps"]) > 0
        assert body["graph_base64"].startswith("data:image/png;base64,")

    def test_linear_solves_correctly(self, client: TestClient) -> None:
        response = client.post("/solve/latex", json={"latex": "2x + 3 = 7"})
        assert response.status_code == 200
        assert "2" in response.json()["answer"]

    def test_invalid_latex_returns_422(self, client: TestClient) -> None:
        response = client.post("/solve/latex", json={"latex": "@@@ not math @@@"})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_missing_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/solve/latex", json={})
        assert response.status_code == 422


class TestExplain:
    def test_explain_returns_structured_fields(self, client: TestClient) -> None:
        response = client.post(
            "/explain",
            json={"latex": "x^2 - 5x + 6 = 0", "level": "class_8_10", "language": "hinglish"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["text"]
        assert body["source"] in ("gemini", "template")

    def test_invalid_level_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/explain", json={"latex": "x^2 - 5x + 6 = 0", "level": "not_a_real_level"}
        )
        assert response.status_code == 400


class TestChat:
    def test_chat_with_context_gives_shortcut(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "session_id": "test_session",
                "message": "Iska shortcut kya hai?",
                "problem_latex": "x^2 - 5x + 6 = 0",
            },
        )
        assert response.status_code == 200
        assert "vieta" in response.json()["text"].lower()

    def test_chat_without_context_asks_for_problem(self, client: TestClient) -> None:
        response = client.post(
            "/chat", json={"session_id": "fresh_session_xyz", "message": "explain karo"}
        )
        assert response.status_code == 200
        assert response.json()["text"]


class TestPractice:
    def test_practice_returns_verified_problem(self, client: TestClient) -> None:
        response = client.get("/practice/quadratic")
        assert response.status_code == 200
        body = response.json()
        assert body["topic"] == "Quadratic Equations"
        assert body["latex_problem"]
        assert body["latex_answer"]


class TestStudentProfile:
    def test_record_and_fetch_roundtrip(self, client: TestClient) -> None:
        record_response = client.post(
            "/student-profile/attempt",
            json={
                "student_id": "pytest_student",
                "topic": "Algebra",
                "correct": True,
                "solve_time_seconds": 42,
                "latex_problem": "2x=4",
            },
        )
        assert record_response.status_code == 200

        profile_response = client.get("/student-profile", params={"student_id": "pytest_student"})
        assert profile_response.status_code == 200
        body = profile_response.json()
        assert body["summary"]["total_solved"] == 1
        assert "Algebra" in body["heatmap"]

    def test_new_student_has_empty_profile(self, client: TestClient) -> None:
        response = client.get("/student-profile", params={"student_id": "brand_new_pytest_student"})
        assert response.status_code == 200
        assert response.json()["summary"]["total_solved"] == 0
