"""Turn LaTeX-flavoured math text into something a student can read or hear.

The step extractor and solver produce strings such as
``x^{2} - 5x + 6 = 0 is a Quadratic Equation`` or
``\\left(x - 3\\right) \\left(x - 2\\right) = 0``. Rendered as plain text those look
broken on screen and are read out letter by letter by a text-to-speech engine.

* :func:`pretty_math` -- readable Unicode text for on-screen cards (``x² - 5x + 6 = 0``).
* :func:`speak_math`  -- words for TTS (``x squared minus 5 x plus 6 equals 0``).

Both are idempotent on text that has no LaTeX in it, so they are safe to apply to
whole sentences.
"""

from __future__ import annotations

import re

_SUPERSCRIPT = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUBSCRIPT = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUP_OK = set("0123456789+-=()ni")
_SUB_OK = set("0123456789+-=()")

_SYMBOLS = {
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\pi": "π", r"\theta": "θ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\phi": "φ", r"\omega": "ω", r"\Delta": "Δ",
    r"\infty": "∞", r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\int": "∫", r"\sum": "Σ", r"\to": "→", r"\rightarrow": "→", r"\circ": "°",
    r"\ldots": "…", r"\dots": "…",
}
_FUNCTIONS = ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "lim", "det", "exp", "max", "min")


def _read_group(text: str, start: int) -> tuple[str, int]:
    """Return ``(content, index_after)`` of the ``{...}`` group opening at ``text[start]``."""
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1: idx], idx + 1
    return text[start + 1:], len(text)


def _replace_macro_with_args(text: str, macro: str, n_args: int, render) -> str:
    """Replace every ``macro{a}{b}...`` using ``render(*args)`` (arguments are processed first)."""
    out: list[str] = []
    i = 0
    while True:
        k = text.find(macro, i)
        if k < 0:
            out.append(text[i:])
            break
        end = k + len(macro)
        if end < len(text) and text[end].isalpha():          # \fractal-style false match
            out.append(text[i:end])
            i = end
            continue
        out.append(text[i:k])
        j, args = end, []
        for _ in range(n_args):
            while j < len(text) and text[j] == " ":
                j += 1
            if j < len(text) and text[j] == "{":
                content, j = _read_group(text, j)
                args.append(_replace_macro_with_args(content, macro, n_args, render))
            else:
                break
        if len(args) != n_args:
            out.append(text[k:end])
            i = end
            continue
        out.append(render(*args))
        i = j
    return "".join(out)


def _wrap(expr: str) -> str:
    """Parenthesise ``expr`` unless it is a single token."""
    expr = expr.strip()
    return expr if re.fullmatch(r"[\w.]+|[^\s]", expr) else f"({expr})"


def _script(text: str, marker: str, table: dict, ok: set, fallback_open: str) -> str:
    """Convert ``^{..}`` / ``_{..}`` (and single-char forms) to Unicode scripts when possible."""
    pattern = re.compile(re.escape(marker) + r"(?:\{([^{}]*)\}|([0-9A-Za-z+\-]))")

    def repl(match: re.Match) -> str:
        body = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        if body and all(ch in ok for ch in body):
            return body.translate(table)
        return f"{marker}{body}" if len(body) == 1 else f"{marker}({body})"

    return pattern.sub(repl, text)


def pretty_math(text: str) -> str:
    """Convert LaTeX fragments inside ``text`` to readable Unicode.

    Args:
        text: Any string, possibly containing LaTeX such as ``x^{2}``, ``\\frac{a}{b}``,
            ``\\left( ... \\right)`` or ``\\sqrt{x}``.

    Returns:
        Text suitable for on-screen display, e.g. ``x² - 5x + 6 = 0``.
    """
    if not text:
        return text
    text = text.replace("[check]", "✓")
    if "\\" not in text and "^" not in text and "_" not in text and "{" not in text:
        return text
    out = text
    out = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", out)
    out = re.sub(r"\\begin\{[a-zA-Z]*matrix\}(.*?)\\end\{[a-zA-Z]*matrix\}",
                 lambda m: "[" + m.group(1).replace("\\\\", "; ").replace("&", " ").strip() + "]", out, flags=re.S)
    out = re.sub(r"\\(?:left|right)\s*([\(\)\[\]|.]|\\\{|\\\})", lambda m: m.group(1).replace("\\", "").replace(".", ""), out)
    out = re.sub(r"\\(?:mathrm|text|textrm|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", out)
    for macro, symbol in sorted(_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(re.escape(macro) + r"(?![A-Za-z])", symbol, out)
    out = _replace_macro_with_args(out, "\\frac", 2, lambda a, b: f"{_wrap(a)}/{_wrap(b)}")
    out = _replace_macro_with_args(out, "\\sqrt", 1, lambda a: f"√{_wrap(a)}")
    out = re.sub(r"\\hat\s*\{?([ijk])\}?", lambda m: {"i": "î", "j": "ĵ", "k": "k̂"}[m.group(1)], out)
    out = re.sub(r"\\vec\s*\{([A-Za-z])\}", lambda m: m.group(1) + "⃗", out)
    for fn in _FUNCTIONS:
        out = re.sub(r"\\" + fn + r"(?![A-Za-z])", fn, out)
    out = re.sub(r"\\[,;:! ]", " ", out)
    out = _script(out, "^", _SUPERSCRIPT, _SUP_OK, "^")
    out = _script(out, "_", _SUBSCRIPT, _SUB_OK, "_")
    out = out.replace("\\", "")
    out = re.sub(r"[{}]", "", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


_SPOKEN_SYMBOLS = {
    "²": " squared ", "³": " cubed ", "√": " square root of ", "±": " plus or minus ", "≈": " approximately ",
    "≤": " less than or equal to ", "≥": " greater than or equal to ", "≠": " not equal to ", "×": " times ",
    "·": " times ", "÷": " divided by ", "π": " pi ", "θ": " theta ", "α": " alpha ", "β": " beta ",
    "∞": " infinity ", "∫": " integral of ", "→": " goes to ", "°": " degrees ", "✓": " checked ",
    "î": " i hat ", "ĵ": " j hat ", "k̂": " k hat ",
}
_VECTOR_ARROW = "⃗"


def speak_math(text: str) -> str:
    """Rewrite ``text`` so a text-to-speech engine reads the maths naturally.

    Args:
        text: Explanation or answer text, possibly with LaTeX or math symbols.

    Returns:
        E.g. ``x^2 - 5x + 6 = 0`` -> ``x squared minus 5 x plus 6 equals 0``.
    """
    out = re.sub(r"\^\s*\{?\s*(?:\\circ|°)\s*\}?", "°", text)
    out = re.sub(r"\^\{?\s*2\s*\}?", " squared ", out)
    out = re.sub(r"\^\{?\s*3\s*\}?", " cubed ", out)
    out = re.sub(r"\^\{?\s*([^{}\s]+)\s*\}?", r" to the power \1 ", out)
    out = pretty_math(out)
    for sym, words in _SPOKEN_SYMBOLS.items():
        out = out.replace(sym, words)
    out = re.sub(r"[⁰¹⁴⁵⁶⁷⁸⁹ⁿ]+", lambda m: " to the power " + m.group(0).translate(
        str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ", "0123456789n")) + " ", out)
    out = re.sub(r"([A-Za-z])" + _VECTOR_ARROW, r"vector \1", out)
    out = out.replace("−", " minus ")
    out = re.sub(r"(?<=[\w)\]])\s+-\s+(?=[\w(√])", " minus ", out)           # a - b
    out = re.sub(r"(?:(?<=[=(,])|(?<=^)|(?<=\s))-(?=\d|[A-Za-z(])", " minus ", out) if re.search(r"[=(]\s*-", out) else out
    out = re.sub(r"\s*=\s*", " equals ", out)
    out = re.sub(r"\s*\+\s*", " plus ", out)
    out = re.sub(r"(?<=[\w)])\s*/\s*(?=[\w(])", " divided by ", out)
    out = re.sub(r"\*", " times ", out)
    out = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", out)                            # 5x -> 5 x
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()
