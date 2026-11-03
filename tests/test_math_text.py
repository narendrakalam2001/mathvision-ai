"""pretty_math (on-screen) and speak_math (text-to-speech) for LaTeX-flavoured step text."""

from __future__ import annotations

import pytest

from src.math_text import pretty_math, speak_math


@pytest.mark.parametrize(("raw", "expected"), [
    ("x^{2} - 5x + 6 = 0", "x² - 5x + 6 = 0"),
    ("ax^2 + bx + c = 0", "ax² + bx + c = 0"),
    (r"\left(x - 3\right) \left(x - 2\right) = 0", "(x - 3) (x - 2) = 0"),
    ("Verify x=2: 0 = 0 [check]", "Verify x=2: 0 = 0 ✓"),
    (r"\frac{a}{b} + \frac{1}{x+1}", "a/b + 1/(x+1)"),
    (r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}", "x = (-b ± √(b²-4ac))/2a"),
    (r"\theta = \frac{\pi}{4} \approx 45.00^\circ", "θ = π/4 ≈ 45.00°"),
    (r"x_{1} + x_{2} = 5", "x₁ + x₂ = 5"),
    (r"\sqrt{49}", "√49"),
    (r"\vec{a} \times \vec{b} = 5\hat{i} - 5\hat{j}", "a⃗ × b⃗ = 5î - 5ĵ"),
    (r"\sin^{2}{\left(x \right)} + \cos^{2}{\left(x \right)}", "sin²(x) + cos²(x)"),
    (r"\int x^{2} dx = \frac{x^{3}}{3} + C", "∫ x² dx = (x³)/3 + C"),
])
def test_pretty_math(raw: str, expected: str) -> None:
    assert pretty_math(raw) == expected


@pytest.mark.parametrize("plain", [
    "", "Class 8-10 step-by-step solution", "Standard form helps us apply the quadratic formula", "ANSWER: x = 2 or x = 3",
])
def test_plain_text_is_untouched(plain: str) -> None:
    assert pretty_math(plain) == plain


def test_pretty_math_is_idempotent() -> None:
    once = pretty_math(r"x^{2} - \frac{a}{b} \le 4")
    assert pretty_math(once) == once


def test_no_latex_leftovers() -> None:
    out = pretty_math(r"\left(x - 3\right) \cdot \frac{\sqrt{x}}{2} \geq \pi")
    assert "\\" not in out and "{" not in out and "}" not in out


@pytest.mark.parametrize(("raw", "expected"), [
    ("x^2 - 5x + 6 = 0", "x squared minus 5 x plus 6 equals 0"),
    ("ANSWER: x = 2 or x = 3", "ANSWER: x equals 2 or x equals 3"),
    (r"\frac{a}{b}", "a divided by b"),
    (r"\sqrt{x+1}", "square root of (x plus 1)"),
    ("45.00^\\circ", "45.00 degrees"),
    ("x^3", "x cubed"),
    ("Verify x=2: 0 = 0 [check]", "Verify x equals 2: 0 equals 0 checked"),
])
def test_speak_math(raw: str, expected: str) -> None:
    assert speak_math(raw) == expected


def test_speak_math_keeps_hyphenated_words() -> None:
    assert "step-by-step" in speak_math("a step-by-step method for x - 1 = 0")


def test_voice_engine_speaks_symbols_as_words() -> None:
    from src.voice_engine import VoiceEngine, VoiceLanguage

    english = VoiceEngine().mathify_text("x^2 = 4", VoiceLanguage.ENGLISH)
    hinglish = VoiceEngine().mathify_text("x^2 = 4", VoiceLanguage.HINGLISH)
    assert english == "x squared equals 4"
    assert "barabar" in hinglish          # the Hindi term table finally gets words to work on
