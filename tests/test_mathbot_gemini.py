"""MathBot: deterministic tools stay exact; free-form doubts go to Gemini (grounded); outages fall back."""

from __future__ import annotations

import dataclasses

import pytest

import src.mathbot as mb
from src.gemini_client import GeminiError


class _NoRAG:
    def query(self, *_a, **_k):
        return []

    def formula_lookup(self, *_a, **_k):
        return None


@pytest.fixture()
def bot(monkeypatch):
    monkeypatch.setattr(mb, "settings", dataclasses.replace(mb.settings, enable_gemini_explanations=True))
    bot = mb.MathBot(gemini_api_key="k", rag_engine=_NoRAG())
    bot.set_context("s1", r"x^2 - 5x + 6 = 0")
    return bot


def test_free_form_doubt_goes_to_gemini_with_the_verified_solution(bot, monkeypatch) -> None:
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"], seen["system"] = prompt, kwargs.get("system")
        return "Discriminant batata hai kitne roots hain. Samjha?"

    monkeypatch.setattr(mb, "generate_text", fake)
    reply = bot.ask("s1", "Main step 3 nahi samjha, discriminant kyun calculate kiya?")
    assert reply.source == "gemini" and "Discriminant" in reply.text
    assert r"x^2 - 5x + 6 = 0" in seen["prompt"]
    assert "VERIFIED FINAL ANSWER" in seen["prompt"] and "x = 2" in seen["prompt"]
    assert "STUDENT'S DOUBT" in seen["prompt"]
    assert "MathGuru" in seen["system"] and "Never contradict" in seen["system"]


@pytest.mark.parametrize("message", ["Iska shortcut kya hai?", "Ek similar problem do", "Graph dikhao"])
def test_tool_intents_never_call_gemini(bot, monkeypatch, message: str) -> None:
    monkeypatch.setattr(mb, "generate_text", lambda *a, **k: pytest.fail("Gemini must not be called"))
    reply = bot.ask("s1", message)
    assert reply.source == "local_fallback" and reply.text


def test_graph_tool_still_returns_an_image(bot, monkeypatch) -> None:
    monkeypatch.setattr(mb, "generate_text", lambda *a, **k: pytest.fail("Gemini must not be called"))
    assert bot.ask("s1", "Graph dikhao").graph_base64


def test_gemini_outage_falls_back_to_local_logic(bot, monkeypatch) -> None:
    def down(*_a, **_k):
        raise GeminiError("503")

    monkeypatch.setattr(mb, "generate_text", down)
    reply = bot.ask("s1", "Step 2 explain karo")
    assert reply.source == "local_fallback" and reply.text


def test_without_api_key_nothing_calls_gemini(monkeypatch) -> None:
    monkeypatch.setattr(mb, "settings", dataclasses.replace(mb.settings, gemini_api_key="", enable_gemini_explanations=True))
    monkeypatch.setattr(mb, "generate_text", lambda *a, **k: pytest.fail("Gemini must not be called"))
    offline = mb.MathBot(gemini_api_key="", rag_engine=_NoRAG())
    offline.set_context("s", r"x^2 - 5x + 6 = 0")
    assert offline.ask("s", "Main step 3 nahi samjha").text


def test_chat_history_is_passed_on_the_next_turn(bot, monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(mb, "generate_text", lambda prompt, **k: prompts.append(prompt) or "ok")
    bot.ask("s1", "pehla sawaal kyun?")
    bot.ask("s1", "aur doosra sawaal?")
    assert "pehla sawaal kyun?" in prompts[1] and "CHAT SO FAR" in prompts[1]
    assert "CHAT SO FAR" not in prompts[0]
