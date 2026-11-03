"""End-to-end UI flows driven headlessly with ``streamlit.testing.v1.AppTest``.

Covers the problems found in manual testing: stats stuck at zero, recents only filled by
"Save to Notebook", Hinglish text in the UI, steps not following the language selector,
and the dashboard practice button showing nothing.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import src.llm_explainer as le  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
HINGLISH = ("karo", "nahi", "sawaal", "jawab", "samjha", "dikhao", "abhi tak", "pehle", "haan", "yeh ")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDENT_PROFILE_DIR", str(tmp_path))
    import src.config as config

    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, student_profile_dir=tmp_path))
    import src.student_profile as sp

    monkeypatch.setattr(sp, "settings", config.settings)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def _nav(at: AppTest, name: str) -> AppTest:
    return [b for b in at.sidebar.button if name in b.label][0].click().run()


def _solve_typed(at: AppTest, latex: str) -> AppTest:
    _nav(at, "Home")
    at.text_input[0].set_value(latex).run()
    at.button[0].click().run()
    return [b for b in at.button if "Yes, solve" in b.label][0].click().run()


def _metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def _cards(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith('<div class="mv-card')]


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [b.label for b in at.button]
    parts += [h.value for h in at.header] + [i.value for i in at.info] + [w.value for w in at.warning]
    return "\n".join(parts).lower()


class TestStatsUpdateOnEverySolve:
    def test_home_starts_empty(self, app) -> None:
        assert _metrics(app) == {"Problems today": "0", "Weak topic": "None yet", "Streak 🔥": "0 days"}

    def test_one_solve_updates_home_without_saving_to_the_notebook(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _nav(app, "Home")
        assert _metrics(app)["Problems today"] == "1" and _metrics(app)["Streak 🔥"] == "1 day"

    def test_problem_shows_in_recent_after_a_normal_solve(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _nav(app, "Home")
        assert any("x² - 5x + 6 = 0" in b.label for b in app.button if b.label.startswith("↻"))

    def test_rerunning_the_solution_screen_does_not_double_count(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        app.run()
        app.run()
        _nav(app, "Dashboard")
        assert _metrics(app)["Total Solved"] == "1"

    def test_two_problems_count_twice(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _solve_typed(app, "2x + 3 = 7")
        _nav(app, "Dashboard")
        assert _metrics(app)["Total Solved"] == "2"

    def test_dashboard_heatmap_and_recent_fill_in(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _nav(app, "Dashboard")
        text = _text(app)
        assert "quadratic equation" in text and "strong" in text and "(100%, 1 solved)" in text
        assert _metrics(app)["Accuracy"] == "100.0%"

    def test_reopening_a_recent_problem_goes_back_to_processing(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _nav(app, "Home")
        [b for b in app.button if b.label.startswith("↻")][0].click().run()
        assert app.session_state["screen"] == "🔍 Processing" and app.session_state["raw_latex"] == "x^2 - 5x + 6 = 0"


class TestSelfAssessmentAndNotebook:
    def test_still_confused_creates_a_weak_topic(self, app) -> None:
        _solve_typed(app, "2x + 3 = 7")
        [b for b in app.button if "confused" in b.label][0].click().run()
        _nav(app, "Home")
        assert _metrics(app)["Weak topic"] == "Linear Equation"
        _nav(app, "Dashboard")
        assert _metrics(app)["Accuracy"] == "0.0%"

    def test_understood_can_be_restored(self, app) -> None:
        _solve_typed(app, "2x + 3 = 7")
        [b for b in app.button if "confused" in b.label][0].click().run()
        [b for b in app.button if "I understood" in b.label][0].click().run()
        _nav(app, "Home")
        assert _metrics(app)["Weak topic"] == "None yet"

    def test_notebook_only_fills_when_saved(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        _nav(app, "Dashboard")
        assert "problems you save" in _text(app)
        _nav(app, "Solution")
        [b for b in app.button if "Save to Notebook" in b.label][0].click().run()
        _nav(app, "Dashboard")
        assert "x² - 5x + 6 = 0" in _text(app).split("my notebook")[1]

    def test_saving_twice_keeps_one_entry(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        for _ in range(2):
            [b for b in app.button if "Save to Notebook" in b.label][0].click().run()
        _nav(app, "Dashboard")
        assert _text(app).split("my notebook")[1].count("x² - 5x + 6 = 0") == 1


class TestDashboardPractice:
    def test_practice_problem_is_shown_on_the_dashboard(self, app) -> None:
        _solve_typed(app, "2x + 3 = 7")
        [b for b in app.button if "confused" in b.label][0].click().run()
        _nav(app, "Dashboard")
        [b for b in app.button if "practice problem" in b.label.lower()][0].click().run()
        assert app.session_state["screen"] == "📊 Dashboard" and app.latex


class TestEnglishUi:
    @pytest.mark.parametrize("screen", ["Home", "Dashboard", "MathBot", "Formulas", "Solution"])
    def test_no_hinglish_on_any_screen(self, app, screen: str) -> None:
        _nav(app, screen)
        text = _text(app)
        assert not [w for w in HINGLISH if w in text], f"Hinglish found on {screen}"

    def test_header_and_sidebar_labels(self, app) -> None:
        assert any("Every question answered — instantly" in m.value for m in app.markdown)
        assert app.sidebar.caption[0].value == "Student · Class 12"

    def test_solved_screens_are_english_too(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        assert not [w for w in HINGLISH if w in _text(app)]
        _nav(app, "Home")
        assert not [w for w in HINGLISH if w in _text(app)]

    def test_chat_chips_are_english(self, app) -> None:
        _nav(app, "MathBot")
        labels = [b.label for b in app.button if b.label not in {s for s in ("🏠 Home",)}]
        assert "Explain step 3" in labels and "Show the graph" in labels


class TestStepsReadableOnScreen:
    def test_no_raw_latex_in_step_cards(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        joined = " ".join(_cards(app))
        assert "\\left" not in joined and "x^{2}" not in joined and "[check]" not in joined
        assert "x²" in joined and "✓" in joined

    def test_answer_box_is_readable(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        assert any("ANSWER: x = 2 or x = 3" in m.value for m in app.markdown)


class TestLanguageSelector:
    @pytest.fixture()
    def with_gemini(self, app, monkeypatch):
        calls = {"n": 0}

        def fake(prompt, **_kw):
            calls["n"] += 1
            items = json.loads(prompt[prompt.index("["):])
            return json.dumps([{"title": "HI:" + i["title"], "content": i["content"], "why": "HI:" + i["why"]}
                               for i in items], ensure_ascii=False)

        monkeypatch.setattr(le, "generate_text", fake)
        monkeypatch.setattr(le, "settings", dataclasses.replace(le.settings, gemini_api_key="k", enable_gemini_explanations=True))
        st.cache_resource.clear()
        return AppTest.from_file(APP, default_timeout=120).run(), calls

    def test_switching_language_rerenders_the_steps(self, with_gemini) -> None:
        at, calls = with_gemini
        _solve_typed(at, "x^2 - 5x + 6 = 0")
        assert not any("HI:" in c for c in _cards(at))
        at.selectbox(key="lang_select").select("हिंदी").run()
        assert all("HI:" in c for c in _cards(at)) and calls["n"] == 1
        assert re.search(r"x² - 5 ?x \+ 6 = 0", _cards(at)[0])            # math survives translation

    def test_translation_is_cached_per_language(self, with_gemini) -> None:
        at, calls = with_gemini
        _solve_typed(at, "x^2 - 5x + 6 = 0")
        for label in ("हिंदी", "Hinglish", "हिंदी", "English", "Hinglish"):
            at.selectbox(key="lang_select").select(label).run()
        assert calls["n"] == 2                                              # one per language, not per switch

    def test_back_to_english_restores_the_original_steps(self, with_gemini) -> None:
        at, _ = with_gemini
        _solve_typed(at, "x^2 - 5x + 6 = 0")
        at.selectbox(key="lang_select").select("हिंदी").run()
        at.selectbox(key="lang_select").select("English").run()
        assert not any("HI:" in c for c in _cards(at))

    def test_without_gemini_the_english_steps_stay_with_a_note(self, app) -> None:
        _solve_typed(app, "x^2 - 5x + 6 = 0")
        app.selectbox(key="lang_select").select("हिंदी").run()
        assert _cards(app) and any("Showing the steps in English" in i.value for i in app.info)

    def test_explanation_regenerates_when_the_language_changes(self, app, monkeypatch) -> None:
        import services.math_service as svc_module

        seen: list[str] = []
        original = svc_module.MathService.explain
        monkeypatch.setattr(svc_module.MathService, "explain",
                            lambda self, latex, level, language: seen.append(language) or original(self, latex, level, language))
        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=120).run()
        _solve_typed(at, "x^2 - 5x + 6 = 0")
        [b for b in at.button if "friendly explanation" in b.label][0].click().run()
        at.selectbox(key="lang_select").select("Hinglish").run()
        assert seen == ["english", "hinglish"]


class TestProfileEditor:
    def test_name_and_class_are_editable(self, app) -> None:
        app.sidebar.text_input[0].set_value("Asha").run()
        app.sidebar.selectbox[0].select("Class 10").run()
        assert app.sidebar.caption[0].value == "Asha · Class 10"
