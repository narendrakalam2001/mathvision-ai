"""FastAPI serving layer — MathVision AI's HTTP API.

Implements the 7 endpoints from the product spec:

    POST /solve               image -> full solution (OCR + solve + steps + graph)
    POST /explain              latex -> level/language-aware explanation
    POST /chat                 doubt-clearing message -> MathBot reply
    POST /voice-question       audio -> transcribed text -> MathBot reply
    GET  /practice/{topic}     topic -> one fresh practice problem
    GET  /student-profile      student_id -> dashboard data
    GET  /health                component availability report

All business logic lives in :mod:`services.math_service`; this module
is purely HTTP glue: request/response models, status codes, and file
upload handling. Errors from the service layer
(:class:`~services.math_service.MathServiceError`) are translated into
clean 4xx responses with a student-facing message rather than a raw
500 stack trace.

Run locally:
    uvicorn serving.api:app --reload --port 8000

Or via the convenience script:
    python scripts/run_api.py
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import threading
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.math_service import MathService, MathServiceError

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Optionally load the OCR model in the background (``PRELOAD_OCR=true``)."""
    from src.config import settings

    if settings.preload_ocr:
        threading.Thread(target=service.warm_up_ocr, name="ocr-warmup", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="MathVision AI API",
    description=(
        "Intelligent handwritten math problem solver & learning assistant. "
        "Every question answered — instantly."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared service instance — model weights / DB connections are
# loaded lazily inside it, so app startup itself stays fast.
service = MathService()




# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class SolveLatexRequest(BaseModel):
    """Body for solving an already-typed/edited LaTeX problem (no image)."""

    latex: str = Field(..., description="The LaTeX problem to solve, e.g. 'x^2 - 5x + 6 = 0'")
    problem_text: Optional[str] = Field(None, description="Optional word-problem narrative")


class ExplainRequest(BaseModel):
    latex: str
    level: str = Field("class_8_10", description="class_8_10 | class_11_12 | jee_advanced")
    language: str = Field("hinglish", description="english | hindi | hinglish")
    style: str = Field("standard", description="standard | visual | story | shortcut")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    problem_latex: Optional[str] = Field(None, description="Sets/refreshes the bot's problem context")
    language: str = Field("english", description="english | hindi | hinglish (language of Gemini replies)")


class SpeakRequest(BaseModel):
    text: str
    language: str = Field("hinglish", description="english | hindi | hinglish")


class RecordAttemptRequest(BaseModel):
    student_id: str
    topic: str
    correct: bool
    solve_time_seconds: float
    latex_problem: str = ""


def _service_error_to_http(exc: MathServiceError) -> HTTPException:
    """Translate a MathServiceError into a clean 422 for the client."""
    return HTTPException(status_code=422, detail=str(exc))


# ----------------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------------
@app.get("/health", summary="Report which optional backends are configured/installed")
def health() -> dict:
    return service.health()


# ----------------------------------------------------------------------
# POST /solve
# ----------------------------------------------------------------------
@app.post("/solve", summary="Solve a handwritten math problem from an uploaded photo")
async def solve_from_image(file: UploadFile = File(...), line_index: int = Form(0)) -> dict:
    """Accepts a JPG/PNG/HEIC image of a handwritten problem.

    Runs the full Module 1 -> 2 -> 3 pipeline and returns the OCR
    result, quality report, problem type, step-by-step solution, and
    (when applicable) a rendered graph. A photo with several lines is
    split per line (``ocr.lines``); ``line_index`` picks the line to solve.
    """
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = service.solve_image(tmp_path, line_index=line_index)
        return result
    except MathServiceError as exc:
        raise _service_error_to_http(exc) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/solve/latex", summary="Solve an already-typed or manually-edited LaTeX problem")
def solve_from_latex(request: SolveLatexRequest) -> dict:
    """Skips OCR entirely — used when the student types a problem directly,
    or edits the OCR'd LaTeX before confirming (Screen 2's edit box)."""
    try:
        return service.solve_latex(request.latex, request.problem_text)
    except MathServiceError as exc:
        raise _service_error_to_http(exc) from exc


# ----------------------------------------------------------------------
# POST /explain
# ----------------------------------------------------------------------
@app.post("/explain", summary="Get a level/language-aware explanation for a solved problem")
def explain(request: ExplainRequest) -> dict:
    try:
        return service.explain(request.latex, request.level, request.language, request.style)
    except MathServiceError as exc:
        raise _service_error_to_http(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid level/language/style: {exc}") from exc


@app.post("/explain/speak", summary="Synthesize a spoken version of explanation text")
def speak(request: SpeakRequest) -> Response:
    try:
        audio_bytes = service.synthesize_speech(request.text, request.language)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except MathServiceError as exc:
        raise _service_error_to_http(exc) from exc


# ----------------------------------------------------------------------
# POST /chat
# ----------------------------------------------------------------------
@app.post("/chat", summary="Send a doubt-clearing message to MathBot")
def chat(request: ChatRequest) -> dict:
    return service.chat(request.session_id, request.message, request.problem_latex, request.language)


# ----------------------------------------------------------------------
# POST /voice-question
# ----------------------------------------------------------------------
@app.post("/voice-question", summary="Ask MathBot a doubt by voice")
async def voice_question(session_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    """Accepts a recorded audio clip (wav/mp3/m4a), transcribes it with
    Whisper, and routes the transcribed text through MathBot."""
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        return service.voice_question(session_id, tmp_path)
    except MathServiceError as exc:
        raise _service_error_to_http(exc) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ----------------------------------------------------------------------
# GET /practice/{topic}
# ----------------------------------------------------------------------
@app.get("/practice/{topic}", summary="Generate one fresh, verified practice problem")
def practice(topic: str) -> dict:
    return service.practice(topic)


# ----------------------------------------------------------------------
# GET /student-profile
# ----------------------------------------------------------------------
@app.get("/student-profile", summary="Fetch a student's dashboard data")
def student_profile(student_id: str) -> dict:
    return service.student_profile(student_id)


@app.post("/student-profile/attempt", summary="Record a solved/attempted problem for a student")
def record_attempt(request: RecordAttemptRequest) -> dict:
    return service.record_attempt(
        request.student_id,
        request.topic,
        request.correct,
        request.solve_time_seconds,
        request.latex_problem,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
