"""Test-session isolation.

A developer's real ``.env`` (Gemini key, Wolfram id, ...) must never leak into the
tests: with a key present, "SymPy failed" paths silently call the live APIs, tests
that expect an error stop raising, and quota is burnt. Empty values set here win over
``.env`` because ``load_dotenv`` does not override variables that already exist.
Student profiles also go to a throw-away directory instead of ``data/student_profiles``.
"""

from __future__ import annotations

import os
import tempfile

for _key in ("GEMINI_API_KEY", "WOLFRAMALPHA_APP_ID", "ELEVENLABS_API_KEY", "HUGGINGFACE_TOKEN"):
    os.environ[_key] = ""
os.environ["PRELOAD_OCR"] = "false"
os.environ["STUDENT_PROFILE_DIR"] = tempfile.mkdtemp(prefix="mathvision_test_profiles_")
