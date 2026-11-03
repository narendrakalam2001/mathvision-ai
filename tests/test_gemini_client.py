"""Tests for src.gemini_client and the Gemini paths built on it (fake client, no network)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
from google.genai import errors  # noqa: E402

import src.gemini_client as gc  # noqa: E402
from src.gemini_client import GeminiError, generate_text  # noqa: E402


class _FakeClient:
    """Mimics ``genai.Client().models.generate_content``; replays scripted outcomes."""

    def __init__(self, outcomes: list) -> None:
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


@pytest.fixture()
def fake(monkeypatch):
    def install(outcomes: list) -> _FakeClient:
        client = _FakeClient(outcomes)
        monkeypatch.setattr(gc, "_client", lambda _key: client)
        monkeypatch.setattr(gc.time, "sleep", lambda _s: None)
        return client

    return install


def test_returns_stripped_text_and_uses_configured_model(fake) -> None:
    client = fake(["  hello  "])
    assert generate_text("hi", api_key="k", model="my-model") == "hello"
    assert client.calls[0]["model"] == "my-model" and client.calls[0]["contents"] == "hi"


def test_default_model_comes_from_settings(fake, monkeypatch) -> None:
    monkeypatch.setattr(gc, "settings", dataclasses.replace(gc.settings, gemini_model="gemini-test-9"))
    client = fake(["ok"])
    generate_text("hi", api_key="k")
    assert client.calls[0]["model"] == "gemini-test-9"


def test_default_model_is_not_the_retired_one() -> None:
    assert "1.5" not in gc.settings.gemini_model


def test_system_instruction_is_forwarded(fake) -> None:
    client = fake(["ok"])
    generate_text("hi", system="be brief", api_key="k")
    assert client.calls[0]["config"].system_instruction == "be brief"


def test_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(gc, "settings", dataclasses.replace(gc.settings, gemini_api_key=""))
    with pytest.raises(GeminiError, match="not configured"):
        generate_text("hi")


def test_model_not_found_is_not_retried_and_hints_at_the_fix(fake) -> None:
    client = fake([errors.ClientError(404, {"error": {"message": "models/x is not found"}})])
    with pytest.raises(GeminiError, match="GEMINI_MODEL"):
        generate_text("hi", api_key="k", retries=3)
    assert len(client.calls) == 1


def test_bad_key_is_not_retried(fake) -> None:
    client = fake([errors.ClientError(403, {"error": {"message": "API key not valid"}})])
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        generate_text("hi", api_key="k", retries=3)
    assert len(client.calls) == 1


def test_transient_error_is_retried_then_succeeds(fake) -> None:
    client = fake([errors.ServerError(503, {"error": {"message": "overloaded"}}), "recovered"])
    assert generate_text("hi", api_key="k", retries=3) == "recovered"
    assert len(client.calls) == 2


def test_gives_up_after_all_retries(fake) -> None:
    fake([errors.ServerError(503, {"error": {"message": "x"}})] * 3)
    with pytest.raises(GeminiError, match="after 3 attempt"):
        generate_text("hi", api_key="k", retries=3)


def test_empty_response_is_an_error(fake) -> None:
    fake(["   "])
    with pytest.raises(GeminiError, match="empty"):
        generate_text("hi", api_key="k", retries=1)


# ---- callers --------------------------------------------------------------
def test_solver_gemini_fallback_uses_wrapper(monkeypatch) -> None:
    import src.math_solver as ms

    monkeypatch.setattr(ms, "generate_text", lambda prompt, **kw: "42\n\nBecause reasons.")
    solver = ms.MathSolver(gemini_api_key="k")
    result = solver._solve_with_gemini("What is 6 times 7?")
    assert result.answer == "42" and result.engine_used == "gemini"


def test_solver_gemini_failure_becomes_solver_error(monkeypatch) -> None:
    import src.math_solver as ms

    def boom(*_a, **_k):
        raise GeminiError("model gone")

    monkeypatch.setattr(ms, "generate_text", boom)
    with pytest.raises(ms.SolverError, match="model gone"):
        ms.MathSolver(gemini_api_key="k")._solve_with_gemini("x")


def test_verification_disagree_and_failure_modes(monkeypatch) -> None:
    import src.math_solver as ms

    solver = ms.MathSolver(gemini_api_key="k")
    monkeypatch.setattr(ms, "generate_text", lambda *a, **k: "DISAGREE")
    assert solver.verify_with_gemini("p", "1") is False
    monkeypatch.setattr(ms, "generate_text", lambda *a, **k: "AGREE")
    assert solver.verify_with_gemini("p", "1") is True

    def boom(*_a, **_k):
        raise GeminiError("down")

    monkeypatch.setattr(ms, "generate_text", boom)
    assert solver.verify_with_gemini("p", "1") is True      # never block the student on an API outage


def test_explainer_wrapper_error_is_a_runtime_error(monkeypatch) -> None:
    import src.llm_explainer as le

    def boom(*_a, **_k):
        raise GeminiError("nope")

    monkeypatch.setattr(le, "generate_text", boom)
    explainer = le.LLMExplainer(gemini_api_key="k")
    monkeypatch.setattr(explainer, "_build_prompt", lambda *a, **k: "prompt")
    with pytest.raises(RuntimeError, match="nope"):
        explainer._explain_with_gemini(SimpleNamespace(steps=[]), None, None, None)
