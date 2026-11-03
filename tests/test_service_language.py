"""Step translation, chat language, blank messages and input errors that must not reach an LLM."""

from __future__ import annotations

import dataclasses
import json

import pytest

import src.llm_explainer as le
import src.math_solver as ms
import src.mathbot as mb
from services.math_service import MathService, MathServiceError
from src.gemini_client import GeminiError
from src.step_extractor import Step


@pytest.fixture(scope="module")
def service() -> MathService:
    return MathService()


@pytest.fixture()
def solved(service):
    return service.solve_latex(r"x^2 - 5x + 6 = 0")


@pytest.fixture()
def gemini_on(monkeypatch, service):
    """Pretend a Gemini key exists (settings is frozen, so swap the module-level object)."""
    monkeypatch.setattr(service.explainer, "gemini_api_key", "k")
    monkeypatch.setattr(le, "settings", dataclasses.replace(le.settings, gemini_api_key="k", enable_gemini_explanations=True))
    monkeypatch.setattr(mb, "settings", dataclasses.replace(mb.settings, gemini_api_key="k", enable_gemini_explanations=True))


def _fake_translation(prompt: str, **_kwargs) -> str:
    items = json.loads(prompt[prompt.index("["):])
    return json.dumps([{"title": "T:" + i["title"], "content": i["content"], "why": "W:" + i["why"]} for i in items],
                      ensure_ascii=False)


class TestStepsInLanguage:
    def test_english_is_returned_unchanged(self, service, solved) -> None:
        out = service.steps_in_language(solved["steps"], "english")
        assert out["steps"] == solved["steps"] and out["translated"] is False and out["note"] == ""

    def test_without_a_key_the_english_steps_come_back_with_a_note(self, service, solved) -> None:
        out = service.steps_in_language(solved["steps"], "hindi")
        assert out["translated"] is False and out["steps"] == solved["steps"] and "Gemini" in out["note"]

    def test_translation_keeps_numbering_math_and_verified_flags(self, service, solved, gemini_on, monkeypatch) -> None:
        monkeypatch.setattr(le, "generate_text", _fake_translation)
        out = service.steps_in_language(solved["steps"], "hindi")
        assert out["translated"] is True and len(out["steps"]) == len(solved["steps"])
        for original, new in zip(solved["steps"], out["steps"]):
            assert new["title"] == "T:" + original["title"]
            assert new["number"] == original["number"] and new["verified"] == original["verified"]
            assert new["content"] == original["content"]            # math untouched

    def test_prompt_demands_unchanged_math_and_names_the_language(self, service, solved, gemini_on, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr(le, "generate_text", lambda prompt, **k: seen.setdefault("p", prompt) and _fake_translation(prompt))
        service.steps_in_language(solved["steps"], "hinglish")
        assert "Hinglish" in seen["p"] and "EXACTLY unchanged" in seen["p"]

    def test_markdown_fenced_json_is_accepted(self, service, solved, gemini_on, monkeypatch) -> None:
        monkeypatch.setattr(le, "generate_text", lambda p, **k: "```json\n" + _fake_translation(p) + "\n```")
        assert service.steps_in_language(solved["steps"], "hindi")["translated"] is True

    @pytest.mark.parametrize("bad", ["not json at all", "[]", '[{"title": "only one"}]', '{"a": 1}'])
    def test_malformed_replies_fall_back_to_english(self, service, solved, gemini_on, monkeypatch, bad) -> None:
        monkeypatch.setattr(le, "generate_text", lambda p, **k: bad)
        out = service.steps_in_language(solved["steps"], "hindi")
        assert out["translated"] is False and out["steps"] == solved["steps"] and out["note"]

    def test_gemini_outage_falls_back_to_english(self, service, solved, gemini_on, monkeypatch) -> None:
        def down(*_a, **_k):
            raise GeminiError("503")

        monkeypatch.setattr(le, "generate_text", down)
        assert service.steps_in_language(solved["steps"], "hindi")["translated"] is False

    def test_unknown_language_is_an_error(self, service, solved) -> None:
        with pytest.raises(MathServiceError):
            service.steps_in_language(solved["steps"], "klingon")

    def test_translate_steps_english_short_circuits(self) -> None:
        steps = [Step(number=1, title="t", content="c", why="w")]
        assert le.LLMExplainer().translate_steps(steps, le.Language.ENGLISH) == steps


class TestChatLanguageAndBlankMessages:
    def test_language_reaches_the_gemini_system_prompt(self, gemini_on, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr(mb, "generate_text", lambda prompt, **k: seen.setdefault("system", k["system"]) and "ok")
        svc = MathService()
        svc.chat("lang1", "Why do we use the discriminant?", problem_latex="x^2 - 5x + 6 = 0", language="hindi")
        assert "Devanagari" in seen["system"]

    def test_default_language_is_english(self, gemini_on, monkeypatch) -> None:
        seen = {}
        monkeypatch.setattr(mb, "generate_text", lambda prompt, **k: seen.setdefault("system", k["system"]) and "ok")
        MathService().chat("lang2", "Why this step?", problem_latex="x^2 - 5x + 6 = 0")
        assert "simple, clear English" in seen["system"]

    def test_blank_message_never_calls_gemini(self, gemini_on, monkeypatch) -> None:
        monkeypatch.setattr(mb, "generate_text", lambda *a, **k: pytest.fail("a blank message must not reach Gemini"))
        reply = MathService().chat("blank", "   ", problem_latex="x^2 - 5x + 6 = 0")
        assert reply["text"] and reply["source"] != "gemini"


class TestOfflineEnglishReplies:
    @pytest.mark.parametrize("message", ["What's the shortcut?", "Give a similar problem", "Show the graph", "Explain step 3"])
    def test_english_questions_are_understood_offline(self, service, message) -> None:
        reply = service.chat("en1", message, problem_latex="x^2 - 5x + 6 = 0")
        assert reply["text"] and "couldn't tell" not in reply["text"]

    def test_show_the_graph_again_still_returns_a_graph(self, service) -> None:
        assert service.chat("en2", "Show the graph again", problem_latex="x^2 - 5x + 6 = 0")["graph_base64"]

    def test_replies_contain_no_hinglish(self, service) -> None:
        for message in ["What's the shortcut?", "Explain step 2", "Give a similar problem"]:
            text = service.chat("en3", message, problem_latex="x^2 - 5x + 6 = 0")["text"].lower()
            assert not any(word in text for word in ("karo", "nahi", "hai!", "dikhao", "kyunki", "samjha"))


class TestInputErrorsSkipLlmFallbacks:
    def test_vector_dimension_mismatch_never_calls_an_llm(self, monkeypatch) -> None:
        monkeypatch.setattr(ms, "settings", dataclasses.replace(ms.settings, gemini_api_key="k", enable_gemini_explanations=True))
        monkeypatch.setattr(ms, "generate_text", lambda *a, **k: pytest.fail("input errors must not reach Gemini"))
        with pytest.raises(MathServiceError, match="same number of components"):
            MathService().solve_latex(r"(1, 2) \cdot (1, 2, 3)")

    def test_solver_input_error_is_a_solver_error(self) -> None:
        assert issubclass(ms.SolverInputError, ms.SolverError)
