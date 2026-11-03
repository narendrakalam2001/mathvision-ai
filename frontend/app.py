"""MathVision AI — Streamlit student app (all 6 screens).

Screens (exactly as specified in the product brief):
    1. Home / Camera            — capture/upload/type a problem
    2. Image Processing         — quality check + editable OCR preview
    3. Solution View            — step-by-step solution, graph, voice, doubt button
    4. MathBot Doubt Clearing   — WhatsApp-style chat with suggested chips
    5. Student Dashboard        — performance overview, topic heatmap, badges
    6. Formula Sheet & Reference — searchable NCERT/JEE formula browser

This app talks directly to :class:`services.math_service.MathService`
(the same orchestration layer used by ``serving/api.py``), so every
screen works standalone without needing the FastAPI server running.
To point the app at a remote deployment of the API instead, swap the
``MathService()`` calls below for ``requests`` calls to the deployed
``/solve``, ``/explain``, etc. endpoints — the response shapes match
exactly, since both layers share the same service code.

Run:
    streamlit run frontend/app.py

Or via the convenience script:
    python scripts/run_app.py
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.math_text import pretty_math
from services.math_service import MathService, MathServiceError  # noqa: E402

# ----------------------------------------------------------------------
# Brand constants (from the product spec)
# ----------------------------------------------------------------------
COLOR_PRIMARY = "#1A237E"
COLOR_SECONDARY = "#FF6F00"
COLOR_SUCCESS = "#2E7D32"
COLOR_BACKGROUND = "#FFFFFF"
COLOR_CARD = "#E3F2FD"

SUBJECTS = ["Algebra", "Geometry", "Trigonometry", "Calculus", "Statistics", "JEE", "NEET"]
SCREENS = [
    "🏠 Home",
    "🔍 Processing",
    "📝 Solution",
    "🤖 MathBot",
    "📊 Dashboard",
    "📚 Formulas",
]

logger = logging.getLogger(__name__)

st.set_page_config(page_title="MathVision AI", page_icon="🧮", layout="centered")


# ----------------------------------------------------------------------
# Cached service singleton (models/DB connections load lazily inside it)
# ----------------------------------------------------------------------
@st.cache_resource
def get_service() -> MathService:
    svc = MathService()
    if settings.preload_ocr:
        threading.Thread(target=svc.warm_up_ocr, name="ocr-warmup", daemon=True).start()
    return svc


service = get_service()


# ----------------------------------------------------------------------
# Brand CSS
# ----------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
        .mv-header {{
            background: {COLOR_PRIMARY}; color: white; padding: 1.2rem 1.5rem;
            border-radius: 12px; margin-bottom: 1rem;
        }}
        .mv-header h1 {{ margin: 0; font-size: 1.6rem; }}
        .mv-header p {{ margin: 0.2rem 0 0 0; opacity: 0.9; }}
        .mv-card {{
            background: {COLOR_CARD}; border-radius: 12px; padding: 1rem 1.2rem;
            margin-bottom: 0.8rem; border-left: 4px solid {COLOR_PRIMARY};
        }}
        .mv-card.verify {{ border-left-color: {COLOR_SUCCESS}; }}
        .mv-answer-box {{
            background: {COLOR_SUCCESS}; color: white; border-radius: 12px;
            padding: 1rem 1.2rem; font-size: 1.1rem; font-weight: 600;
        }}
        .mv-badge {{
            display: inline-block; background: {COLOR_SECONDARY}; color: white;
            border-radius: 20px; padding: 0.3rem 0.9rem; margin: 0.2rem;
            font-size: 0.85rem; font-weight: 600;
        }}
        .mv-chip {{
            display: inline-block; background: white; border: 1.5px solid {COLOR_PRIMARY};
            color: {COLOR_PRIMARY}; border-radius: 20px; padding: 0.3rem 0.9rem;
            margin: 0.2rem; font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="mv-header"><h1>🧮 {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
LEVELS = {"Class 8-10": "class_8_10", "Class 11-12": "class_11_12", "JEE/Advanced": "jee_advanced"}
LANGUAGES = {"English": "english", "हिंदी": "hindi", "Hinglish": "hinglish"}
CLASS_LEVELS = ["Class 8", "Class 9", "Class 10", "Class 11", "Class 12"]
# Which problem types belong to which subject chip on the Home screen (JEE / NEET show everything).
SUBJECT_KEYWORDS = {
    "Algebra": ("linear", "quadratic", "polynomial", "simultaneous", "expression", "arithmetic"),
    "Geometry": ("geometry", "vector", "matrix"),
    "Trigonometry": ("trigonometry",),
    "Calculus": ("differentiation", "integration", "limit"),
    "Statistics": ("probability", "statistics"),
}


def init_state() -> None:
    defaults = {
        "screen": SCREENS[0],
        "student_id": "demo_student",
        "student_name": "Student",
        "student_class": "Class 12",
        "raw_latex": "",
        "quality_report": None,
        "ocr_confidence": None,
        "ocr_lines": [],
        "solve_error": None,
        "solved": None,
        "explanation": None,
        "explanation_wanted": False,
        "explanations": {},
        "step_translations": {},
        "attempt_recorded": True,          # nothing pending until a problem is solved
        "understood": True,
        "practice_problem": None,
        "chat_history": [],
        "solve_start_time": None,
        "level": "class_8_10",
        "language": "english",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(screen: str) -> None:
    st.session_state["screen"] = screen


def _topic_label(problem_type: str) -> str:
    """``quadratic_equation`` -> ``Quadratic Equation``."""
    return problem_type.replace("_", " ").title()


def _on_new_solution() -> None:
    """Reset everything that belongs to the previous problem."""
    st.session_state["attempt_recorded"] = False     # the Solution screen records it once
    st.session_state["explanation_wanted"] = False
    st.session_state["explanation"] = None
    st.session_state["practice_problem"] = None
    st.session_state["understood"] = True


def _record_attempt_once(solved: dict) -> None:
    """Count the problem in the student's stats the first time its solution is shown.

    Previously this only happened on "Save to Notebook", so Home, the streak and the
    Dashboard stayed at zero. A failure here must never break the Solution screen.
    """
    if st.session_state.get("attempt_recorded"):
        return
    st.session_state["attempt_recorded"] = True
    elapsed = time.time() - (st.session_state.get("solve_start_time") or time.time())
    try:
        result = service.record_attempt(
            st.session_state["student_id"], _topic_label(solved["problem_type"]), True, elapsed, solved["latex"]
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not record the attempt", exc_info=True)
        return
    for badge in result.get("newly_earned_badges", []):
        st.toast(f"🏆 New badge earned: {badge}")


# ----------------------------------------------------------------------
# Screen 1 — Home / Camera
# ----------------------------------------------------------------------
def screen_home() -> None:
    header("MathVision AI", "Every question answered — instantly")

    profile = service.student_profile(st.session_state["student_id"])
    summary = profile["summary"]

    cols = st.columns(3)
    cols[0].metric("Problems today", summary["solved_today"])
    weak = profile["weak_topics"][0] if profile["weak_topics"] else "None yet"
    cols[1].metric("Weak topic", weak)
    days = summary["streak_days"]
    cols[2].metric("Streak 🔥", f"{days} day" if days == 1 else f"{days} days")

    st.markdown("### 📸 Solve a Problem")
    tab_camera, tab_upload, tab_type = st.tabs(["Camera", "Upload", "Type Equation"])

    with tab_camera:
        photo = st.camera_input("Take a photo of your problem")
        if photo is not None:
            _handle_image_input(photo.getvalue())

    with tab_upload:
        uploaded = st.file_uploader("Drag & drop an image", type=["jpg", "jpeg", "png", "heic"])
        if uploaded is not None:
            _handle_image_input(uploaded.getvalue())

    with tab_type:
        typed = st.text_input("Type your equation (LaTeX or plain)", placeholder="x^2 - 5x + 6 = 0")
        if st.button("Solve it ✓", type="primary", key="type_solve_btn"):
            if typed.strip():
                _start_typed_problem(typed.strip())
            else:
                st.warning("Please type an equation first.")

    st.markdown("### Recent Problems")
    subject = st.pills("Subject filter", SUBJECTS, selection_mode="single", key="subject_filter",
                       label_visibility="collapsed")
    recent = profile["recent_solutions"]
    if subject in SUBJECT_KEYWORDS:
        keywords = SUBJECT_KEYWORDS[subject]
        recent = [r for r in recent if any(k in r["topic"].lower() for k in keywords)]
    recent = recent[:5]
    if not recent:
        st.caption("Nothing here yet. Solve a problem and it will show up here." if not subject
                   else f"No {subject} problems solved yet.")
    for i, r in enumerate(recent):
        mark = "✅" if r["correct"] else "🤔"
        if st.button(f"↻  {pretty_math(r['latex_problem'])}   ·   {r['topic']} {mark}", key=f"recent_{i}",
                     width="stretch"):
            _start_typed_problem(r["latex_problem"])


def _start_typed_problem(latex: str) -> None:
    """Send a typed (or re-opened) problem to the Processing screen."""
    st.session_state["raw_latex"] = latex
    st.session_state["quality_report"] = None
    st.session_state["ocr_confidence"] = None
    st.session_state["ocr_lines"] = []
    st.session_state["solve_error"] = None
    st.session_state["solved"] = None
    st.session_state["solve_start_time"] = time.time()
    goto("🔍 Processing")
    st.rerun()


def _handle_image_input(image_bytes: bytes) -> None:
    """Store raw image bytes and jump to the Processing screen."""
    st.session_state["_pending_image_bytes"] = image_bytes
    st.session_state["solve_start_time"] = time.time()
    goto("🔍 Processing")
    st.rerun()


# ----------------------------------------------------------------------
# Screen 2 — Image Processing
# ----------------------------------------------------------------------
def _store_ocr_result(result: dict) -> None:
    """Keep OCR text, confidence and (if solved) the solution in session state."""
    ocr = result["ocr"]
    st.session_state["raw_latex"] = ocr["latex"]
    st.session_state["ocr_confidence"] = ocr["confidence"]
    st.session_state["ocr_lines"] = ocr.get("lines", [])
    st.session_state["_applied_line"] = ocr.get("selected_line", 0)
    st.session_state.pop("ocr_line_choice", None)      # new photo -> reset the line picker
    st.session_state["quality_report"] = result["quality"]
    st.session_state["solved"] = None if result.get("solve_error") else result
    st.session_state["solve_error"] = result.get("solve_error")


def screen_processing() -> None:
    header("Reading your problem...", "🔍 Analyzing → ✂️ Cropping → 📖 Reading → 🧮 Solving")

    image_bytes = st.session_state.get("_pending_image_bytes")
    if image_bytes:
        st.image(image_bytes, caption="Original photo", width="stretch")
        if st.session_state.get("ocr_confidence") is None:
            with st.spinner("📖 Reading handwriting..."):
                try:
                    result = service.solve_image(image_bytes)
                except MathServiceError as exc:
                    st.error(str(exc))
                    return

            if result.get("retake_recommended"):
                st.warning(f"⚠️ {result['quality']['message']}")
                st.session_state["_pending_image_bytes"] = None
                if st.button("Retake photo"):
                    goto("🏠 Home")
                    st.rerun()
                return

            _store_ocr_result(result)
    elif not st.session_state.get("raw_latex"):
        st.info("Nothing to process yet. Go to Home and take a photo or type an equation.")
        return

    lines = st.session_state.get("ocr_lines") or []
    if len(lines) > 1:
        st.markdown("### Which problem should I solve?")
        picked = st.radio(
            "Detected lines", list(range(len(lines))), key="ocr_line_choice",
            format_func=lambda i: f"Line {i + 1}: {pretty_math(lines[i]['latex'])[:40]}", label_visibility="collapsed",
        )
        if picked != st.session_state.get("_applied_line", 0):
            st.session_state["_applied_line"] = picked
            st.session_state["raw_latex"] = lines[picked]["latex"]
            st.session_state["ocr_confidence"] = lines[picked]["confidence"]
            st.session_state["solved"] = None
            st.session_state["solve_error"] = None
            st.rerun()
    if st.session_state.get("solve_error"):
        st.warning("The handwriting was read, but the problem could not be solved. "
                   f"Please check the LaTeX and edit it if needed. ({st.session_state['solve_error']})")

    st.markdown("### Recognized Equation")
    band = "🟢" if (st.session_state.get("ocr_confidence") or 1.0) >= 0.6 else "🟡"
    st.latex(st.session_state["raw_latex"].replace("\\\\", " "))
    if st.session_state.get("ocr_confidence") is not None:
        st.caption(f"{band} Confidence: {st.session_state['ocr_confidence']:.0%} — Is this correct?")

    edited = st.text_area("Edit the LaTeX manually if the reading is wrong", value=st.session_state["raw_latex"])

    if st.button("Yes, solve it! ✓", type="primary"):
        st.session_state["raw_latex"] = edited
        with st.spinner("🧮 Solving the equation..."):
            try:
                st.session_state["solved"] = service.solve_latex(edited)
            except MathServiceError as exc:
                st.session_state["solve_error"] = str(exc)
                st.error(str(exc))
                return
        st.session_state["solve_error"] = None
        _on_new_solution()
        st.session_state["_pending_image_bytes"] = None
        goto("📝 Solution")
        st.rerun()


# ----------------------------------------------------------------------
# Screen 3 — Solution View
# ----------------------------------------------------------------------
def _steps_for_display(solved: dict) -> tuple[list[dict], str]:
    """Return the steps in the selected language (translated with Gemini when needed)."""
    language = st.session_state["language"]
    if language == "english":
        return solved["steps"], ""
    cache = st.session_state["step_translations"]
    key = f"{solved['latex']}|{language}"
    if key not in cache:
        with st.spinner("Translating the steps..."):
            data = service.steps_in_language(solved["steps"], language)
        if not data["translated"]:          # do not cache failures: the next rerun may succeed
            return data["steps"], data["note"]
        cache[key] = data
    return cache[key]["steps"], ""


def _render_steps(steps: list[dict]) -> None:
    for step in steps:
        css_class = "mv-card verify" if step.get("verified") else "mv-card"
        st.markdown(
            f'<div class="{css_class}"><b>Step {step["number"]}: {pretty_math(step["title"])}</b><br>'
            f'{pretty_math(step["content"])}<br><i>💡 Why? {pretty_math(step["why"])}</i></div>',
            unsafe_allow_html=True,
        )


def _narration(solved: dict, steps: list[dict]) -> str:
    """Text read aloud by "Listen": the steps in the selected language, then the answer."""
    parts = [f"Problem: {solved['latex']}."]
    parts += [f"Step {s['number']}. {s['title']}. {s['content']}." for s in steps]
    parts.append(f"Answer: {solved['answer']}.")
    return " ".join(parts)[:4500]


def screen_solution() -> None:
    header("Solution", "📐 Step-by-step, verified by SymPy")

    solved = st.session_state.get("solved")
    if not solved:
        st.info("No solved problem yet. Start from the Home screen.")
        if st.button("← Back to Home"):
            goto("🏠 Home")
            st.rerun()
        return

    _record_attempt_once(solved)

    stars = "⭐" * solved["difficulty"] + "☆" * (3 - solved["difficulty"])
    st.markdown(f"**Recognized:** `{pretty_math(solved['latex'])}`")
    st.markdown(f"**Problem type:** 📐 {_topic_label(solved['problem_type'])}  |  **Difficulty:** {stars}")

    col1, col2 = st.columns(2)
    with col1:
        level_label = st.selectbox("Level", list(LEVELS), key="level_select")
        st.session_state["level"] = LEVELS[level_label]
    with col2:
        lang_label = st.selectbox("Language", list(LANGUAGES), key="lang_select")
        st.session_state["language"] = LANGUAGES[lang_label]

    if solved.get("graph_base64"):
        st.markdown("#### Graph")
        st.image(solved["graph_base64"], width="stretch")

    st.markdown("#### Step-by-Step Solution")
    steps, note = _steps_for_display(solved)
    if note:
        st.info(f"Showing the steps in English. {note}")
    _render_steps(steps)

    st.markdown(f'<div class="mv-answer-box">✅ ANSWER: {pretty_math(solved["answer"])}</div>', unsafe_allow_html=True)
    st.write("")

    if st.button("Generate friendly explanation 💬"):
        st.session_state["explanation_wanted"] = True

    explanation = None
    if st.session_state.get("explanation_wanted"):      # re-generated automatically when level/language change
        cache = st.session_state["explanations"]
        key = f"{solved['latex']}|{st.session_state['level']}|{st.session_state['language']}"
        if key not in cache:
            with st.spinner("Generating explanation..."):
                cache[key] = service.explain(solved["latex"], st.session_state["level"], st.session_state["language"])
        explanation = cache[key]
    if explanation:
        st.markdown("#### MathGuru's Explanation")
        st.write(pretty_math(explanation["text"]))
        if explanation.get("common_mistakes"):
            st.markdown("**⚠️ Common mistakes:**")
            for m in explanation["common_mistakes"]:
                st.markdown(f"- {pretty_math(m)}")
        if explanation.get("memory_trick"):
            st.info(f"🧠 **Memory trick:** {pretty_math(explanation['memory_trick'])}")
        if explanation.get("practice_problem"):
            st.success(f"📝 **Try this next:** {pretty_math(explanation['practice_problem'])}")

    st.markdown("#### Actions")
    action_cols = st.columns(4)
    if action_cols[0].button("🔊 Listen"):
        try:
            audio_bytes = service.synthesize_speech(
                explanation["text"] if explanation else _narration(solved, steps), st.session_state["language"]
            )
            st.audio(audio_bytes, format="audio/mp3")
        except MathServiceError as exc:
            st.warning(f"Voice is unavailable right now: {exc}")

    if action_cols[1].button("📝 Practice Similar"):
        st.session_state["practice_problem"] = service.practice(solved["latex"])

    if action_cols[2].button("💬 Ask Doubt"):
        service.chat("streamlit_session", "", problem_latex=solved["latex"])
        goto("🤖 MathBot")
        st.rerun()

    if action_cols[3].button("⭐ Save to Notebook"):
        added = service.save_to_notebook(
            st.session_state["student_id"], solved["latex"], _topic_label(solved["problem_type"]), solved["answer"]
        )
        st.toast("Saved to your notebook ⭐" if added else "Already in your notebook.")

    st.caption("How did it go?")
    feedback = st.columns(2)
    if feedback[0].button("👍 I understood", disabled=st.session_state["understood"]):
        service.mark_last_attempt(st.session_state["student_id"], True)
        st.session_state["understood"] = True
        st.rerun()
    if feedback[1].button("🤔 I'm still confused", disabled=not st.session_state["understood"]):
        service.mark_last_attempt(st.session_state["student_id"], False)
        st.session_state["understood"] = False
        st.toast("Noted. We will suggest more practice on this topic.")
        st.rerun()

    _render_practice_problem()


def _render_practice_problem() -> None:
    if st.session_state.get("practice_problem"):
        p = st.session_state["practice_problem"]
        st.markdown("#### 📝 Practice Problem")
        st.latex(p["latex_problem"])
        with st.expander("Show Answer"):
            st.latex(p["latex_answer"])


# ----------------------------------------------------------------------
# Screen 4 — MathBot Doubt Clearing
# ----------------------------------------------------------------------
def screen_mathbot() -> None:
    header("MathGuru 🤖", "Doubt clearing chatbot")

    solved = st.session_state.get("solved")
    if solved:
        st.caption(f"Context: `{pretty_math(solved['latex'])}`")
    else:
        st.caption("No problem selected yet. Solve one first for the best answers.")
    st.caption(f"Replying in: {next(k for k, v in LANGUAGES.items() if v == st.session_state['language'])} "
               "(change it on the Solution screen)")

    chips = service.mathbot.suggested_chips()
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if chip_cols[i].button(chip, key=f"chip_{i}"):
            _send_chat_message(chip)

    for role, text in st.session_state["chat_history"]:
        if role == "bot_image":
            with st.chat_message("assistant"):
                st.image(text, caption="Graph", width="stretch")
            continue
        with st.chat_message("user" if role == "user" else "assistant"):
            st.write(text)

    user_message = st.chat_input("Type your doubt...")
    if user_message:
        _send_chat_message(user_message)
        st.rerun()


def _send_chat_message(message: str) -> None:
    solved = st.session_state.get("solved")
    result = service.chat(
        "streamlit_session", message, problem_latex=solved["latex"] if solved else None,
        language=st.session_state["language"],
    )
    st.session_state["chat_history"].append(("user", message))
    st.session_state["chat_history"].append(("bot", result["text"]))
    if result.get("graph_base64"):
        st.session_state["chat_history"].append(("bot_image", result["graph_base64"]))


# ----------------------------------------------------------------------
# Screen 5 — Student Dashboard
# ----------------------------------------------------------------------
def screen_dashboard() -> None:
    header("Your Progress", "📊 Performance overview")

    profile = service.student_profile(st.session_state["student_id"])
    summary = profile["summary"]

    cols = st.columns(4)
    cols[0].metric("Total Solved", summary["total_solved"])
    cols[1].metric("This Week", summary["solved_this_week"])
    cols[2].metric("Avg Solve Time", f"{summary['average_solve_time_seconds']}s")
    cols[3].metric("Accuracy", f"{summary['accuracy_pct']}%")
    st.caption("Accuracy is self-assessed: a problem counts as understood unless you mark it "
               "\"I'm still confused\" on the Solution screen.")

    st.markdown("### Topic Heatmap")
    if not profile["heatmap"]:
        st.caption("No topic data yet. Solve some problems to see your strengths.")
    else:
        strength_emoji = {"Strong": "🟩🟩🟩🟩", "Good": "🟩🟩🟩", "Need Practice": "🟨🟨", "Weak": "🟥"}
        for topic, stats in profile["heatmap"].items():
            bar = strength_emoji.get(stats["strength"], "⬜")
            st.markdown(f"**{topic}**: {bar} {stats['strength']} ({stats['accuracy']:.0%}, {stats['attempts']} solved)")

    if profile["weak_topics"]:
        st.warning(f"⚠️ Keep practising: {', '.join(profile['weak_topics'])}")
        if st.button(f"Start a {profile['weak_topics'][0]} practice problem"):
            st.session_state["practice_problem"] = service.practice(profile["weak_topics"][0])
    _render_practice_problem()

    st.markdown("### Achievement Badges")
    if summary["badges"]:
        st.markdown(" ".join(f'<span class="mv-badge">🏆 {b}</span>' for b in summary["badges"]), unsafe_allow_html=True)
    else:
        st.caption("No badges yet. Keep solving!")

    st.markdown("### Recent Solutions")
    if not profile["recent_solutions"]:
        st.caption("Your solved problems will appear here.")
    for r in profile["recent_solutions"][:10]:
        st.markdown(f"- `{pretty_math(r['latex_problem'])}` — {r['topic']} — {r['timestamp'][:16]} "
                    f"{'✅' if r['correct'] else '🤔'}")

    st.markdown("### My Notebook")
    if not profile["notebook"]:
        st.caption("Problems you save with ⭐ Save to Notebook appear here.")
    for entry in profile["notebook"][:20]:
        st.markdown(f"- `{pretty_math(entry['latex_problem'])}` → {pretty_math(entry['answer'])} "
                    f"({entry['topic']}, {entry['saved_at'][:10]})")


# ----------------------------------------------------------------------
# Screen 6 — Formula Sheet & Reference
# ----------------------------------------------------------------------
def screen_formulas() -> None:
    header("Formula Sheet", "📚 NCERT & JEE reference")

    import json

    formula_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "formula_sheet.json"
    formula_sheet = json.loads(formula_path.read_text(encoding="utf-8"))

    search = st.text_input("🔍 Search formulas", placeholder="e.g. quadratic formula")

    category_labels = {
        "algebra": "Algebra", "trigonometry": "Trigonometry", "calculus": "Calculus",
        "coordinate_geometry": "Coordinate Geometry", "statistics_probability": "Statistics & Probability",
        "matrices_vectors": "Matrices & Vectors",
    }
    tabs = st.tabs(list(category_labels.values()))

    for tab, (category, label) in zip(tabs, category_labels.items()):
        with tab:
            formulas = formula_sheet.get(category, [])
            if search:
                formulas = [f for f in formulas if search.lower() in f["name"].lower() or search.lower() in f["description"].lower()]
            if not formulas:
                st.caption("No formula found.")
            for formula in formulas:
                with st.container():
                    st.markdown(f"**{pretty_math(formula['name'])}** ({formula['class_level']})")
                    st.latex(formula["latex"])
                    st.caption(pretty_math(formula["description"]))
                    st.divider()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    init_state()
    inject_css()

    with st.sidebar:
        st.markdown("## 🧮 MathVision AI")
        st.caption(f"{st.session_state['student_name']} · {st.session_state['student_class']}")
        with st.expander("Edit profile"):      # widgets bound to session_state keys: the caption above updates at once
            st.text_input("Name", key="student_name")
            st.selectbox("Class", CLASS_LEVELS, key="student_class")
        for screen in SCREENS:
            if st.button(screen, width="stretch", key=f"nav_{screen}"):
                goto(screen)
                st.rerun()

        st.divider()
        health = service.health()
        st.caption("**Backend status**")
        for key, value in health.items():
            if key == "status":
                continue
            icon = "✅" if value else "⚪"
            st.caption(f"{icon} {key.replace('_', ' ')}")

    screen = st.session_state["screen"]
    if screen == "🏠 Home":
        screen_home()
    elif screen == "🔍 Processing":
        screen_processing()
    elif screen == "📝 Solution":
        screen_solution()
    elif screen == "🤖 MathBot":
        screen_mathbot()
    elif screen == "📊 Dashboard":
        screen_dashboard()
    elif screen == "📚 Formulas":
        screen_formulas()


if __name__ == "__main__":
    main()
