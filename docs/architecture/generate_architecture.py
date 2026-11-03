"""Generate ``docs/architecture.svg`` (and ``architecture.png``) for MathVision AI.

The diagram is data-driven: edit the LAYERS list below, run

    python docs/generate_architecture.py

and both files are rebuilt. Every text line is measured with a real font
(DejaVu Sans, wider than Segoe UI/Arial) and the script aborts if a line would
overflow its box, so the SVG never has clipped text.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent
WIDTH = 1100
PAD_X = 18          # inner horizontal padding of a layer band
GAP = 16            # horizontal gap between boxes (holds the arrow)
TITLE_SIZE, SUB_SIZE = 12, 10


@dataclass
class Tone:
    """Colour set of one layer."""
    band: str
    box: str
    stroke: str
    head: str
    title: str
    sub: str
    arrow: str


TONES = {
    "navy":   Tone("#0d1b4b", "#1a3a7a", "#4a7fd4", "#7eb3f7", "#e8f0fe", "#93b8f7", "#4a7fd4"),
    "purple": Tone("#1e0e45", "#2d1070", "#8b5cf6", "#c4a8ff", "#f1e9ff", "#b69cff", "#8b5cf6"),
    "green":  Tone("#0a2e22", "#065f46", "#10b981", "#6ee7b7", "#d1fae5", "#6ee7b7", "#10b981"),
    "cyan":   Tone("#082f3d", "#0e4d63", "#22d3ee", "#67e8f9", "#cffafe", "#7be0f2", "#22d3ee"),
    "orange": Tone("#3d1f08", "#5c2d06", "#f59e0b", "#fcd34d", "#fef3c7", "#f5b84a", "#f59e0b"),
    "maroon": Tone("#2b1114", "#5a0b10", "#dc2626", "#fca5a5", "#fee2e2", "#f4a3a3", "#dc2626"),
}


@dataclass
class Box:
    title: str
    lines: list[str]
    emphasis: bool = False


@dataclass
class Layer:
    name: str
    tone: str
    rows: list[list[Box]]
    support_rows: tuple[int, ...] = ()      # row indexes drawn without flow arrows


B = Box
LAYERS: list[Layer] = [
    Layer("TRAINING & DATA — OFFLINE (KAGGLE T4 GPU)", "navy", [[
        B("CROHME + Symbols", ["~22k handwritten expr.", "19 symbol classes × 150"]),
        B("InkML → Image", ["384×96 canvas, white bg", "3 px strokes, 10 px pad"]),
        B("Synthetic NCERT Set", ["SymPy generator · 9 topics", "solver eval + practice"]),
        B("Fine-tune TrOCR", ["4 epochs · Adafactor", "fp16 · 1 T4 · 5.7 h"]),
        B("Held-out Evaluation", ["1,243 images (5 % split)", "CER 0.138 · exact 82 %"]),
        B("Weights on HF Hub", ["model.safetensors", "+ decoding.json marker"]),
    ]]),
    Layer("MODULES 1–2 — IMAGE PREPROCESSING & HANDWRITING OCR", "purple", [[
        B("Capture / Upload", ["Camera · JPG/PNG/HEIC", "EXIF auto-rotate"]),
        B("Quality Gate", ["Ink-region blur check", "Dark / glare → retake"]),
        B("Preprocess", ["Perspective fix · CLAHE", "Otsu · crop to ink"]),
        B("Layout Analyzer", ["Split page into lines", "Normalise to 384×96"]),
        B("TrOCR-large (tuned)", ["Shifted greedy decode", "KV-cache: 2 tokens/pass"]),
        B("Post-process", ["Symbol fixes · LaTeX", "check · token confidence"]),
    ], [
        B("Pix2Tex Fallback (optional)", ["Only for low-confidence lines", "when the package is installed"]),
        B("Confidence Band", ["Mean token probability →", "green / yellow flag per line"]),
        B("Edit-before-Solve UI", ["Student confirms or fixes", "LaTeX · picks the line"]),
    ]]),
    Layer("MODULE 3 — MATH SOLVER (EXACT, SYMBOLIC FIRST)", "green", [[
        B("LaTeX Parser", ["CROHME-style normaliser", "matrices · systems · e, π"]),
        B("Problem Classifier", ["16 problem types", "rule-based routing"]),
        B("SymPy Solver", ["solve · diff · ∫ · limit", "matrix · y = f(x)"], True),
        B("Fallbacks (optional)", ["WolframAlpha · Gemini", "word problems"]),
        B("Step Extractor", ["Readable steps + “why”", "per-step explanation"]),
        B("Graph Plotter", ["Matplotlib → base64", "lines · parabolas · calculus"]),
    ], [
        B("Vector Engine", ["dot · cross · magnitude · angle", "worked steps"]),
        B("Input Guard", ["Rejects text-like junk", "asks student to edit LaTeX"]),
        B("Answer Verification", ["Substitute back into equation", "optional Gemini cross-check"]),
    ]], support_rows=(1,)),
    Layer("MODULES 4–6 — TEACHING ENGINE", "cyan", [[
        B("LLM Explainer", ["Gemini · 3 levels · 4 styles", "offline template fallback"]),
        B("Voice Engine (TTS)", ["gTTS · Hindi math terms", "ElevenLabs optional"]),
        B("Speech-to-Text", ["Whisper: voice doubts", "Hindi · English"]),
        B("MathBot", ["Tools: graph · practice", "formula · shortcut"], True),
        B("RAG Knowledge Base", ["ChromaDB · formula sheet", "NCERT notes (ingest script)"]),
    ], [
        B("Practice Generator", ["7 topics, answers verified", "by the SymPy solver"]),
        B("Student Profile", ["Weak topics · streaks", "4 achievement badges"]),
        B("Offline-first Design", ["Every module works without", "API keys; APIs only enrich"]),
    ]], support_rows=(1,)),
    Layer("SERVING & FRONTEND", "orange", [[
        B("MathService", ["Orchestrates the pipeline", "solve_image · solve_latex"]),
        B("FastAPI Server", ["10 endpoints: /solve /chat", "/explain /voice-question…"]),
        B("Streamlit App", ["6 screens, mobile-friendly", "Home → Solution → Dashboard"]),
        B("OCR Warm-up", ["PRELOAD_OCR option", "KV-cache self-check"]),
        B("Config & Logging", [".env settings · retries", "friendly Hinglish errors"]),
    ]]),
    Layer("QUALITY, CI/CD & DEPLOYMENT", "maroon", [[
        B("312 pytest Tests", ["unit · pipeline · API", "Gemini via fake client"]),
        B("Solver Evaluation", ["350/350 symbolic solved", "CI gate: ≥ 95 %"]),
        B("GitHub Actions", ["Python 3.11 · pytest", "solver eval · Docker build"]),
        B("Docker", ["Slim image by default", "OCR weights: BuildKit secret"]),
        B("Render / Streamlit Cloud", ["Typed solving: free tier", "Photo OCR needs ≥ 4 GB RAM"]),
    ]]),
]

FOOTER_CHIPS = [
    ("Corpus CER 0.138 (target 0.08 not met)", "#7eb3f7", "#4a7fd4"),
    ("OCR decode 9.2 s → 4.5 s per line (KV-cache)", "#c4a8ff", "#8b5cf6"),
    ("312 tests · Python 3.11", "#6ee7b7", "#10b981"),
    ("English · Hindi · Hinglish voice + text", "#fcd34d", "#f59e0b"),
]


# ----------------------------------------------------------------------
# Text measurement (real font) so nothing overflows
# ----------------------------------------------------------------------
def _fonts():
    from PIL import ImageFont

    base = Path("/usr/share/fonts/truetype/dejavu")
    regular, bold = base / "DejaVuSans.ttf", base / "DejaVuSans-Bold.ttf"
    if not regular.exists():
        return None
    return {
        "title": ImageFont.truetype(str(bold), TITLE_SIZE),
        "sub": ImageFont.truetype(str(regular), SUB_SIZE),
    }


FONTS = _fonts()


def _text_width(text: str, kind: str) -> float:
    if FONTS is None:                                   # fall back to a pessimistic estimate
        return len(text) * (TITLE_SIZE if kind == "title" else SUB_SIZE) * 0.62
    return FONTS[kind].getlength(text)


def _check_fit(box: Box, width: float, layer: str) -> None:
    limit = width - 12
    for kind, text in [("title", box.title)] + [("sub", s) for s in box.lines]:
        if _text_width(text, kind) > limit:
            sys.exit(f"[overflow] layer '{layer}', box '{box.title}': {text!r} "
                     f"is {_text_width(text, kind):.0f}px > {limit:.0f}px")


# ----------------------------------------------------------------------
# SVG building
# ----------------------------------------------------------------------
def _marker(idx: int, color: str) -> str:
    return (f'<marker id="a{idx}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
            f'orient="auto"><path d="M2 2L8 5L2 8" fill="none" stroke="{color}" stroke-width="1.5"/></marker>')


def build_svg() -> str:
    parts: list[str] = []
    markers: list[str] = []
    y = 38
    layer_geometry: list[tuple[int, int]] = []
    for li, layer in enumerate(LAYERS):
        tone = TONES[layer.tone]
        markers.append(_marker(li, tone.arrow))
        row_heights = [34 + 14 * max(len(b.lines) for b in row) + 6 for row in layer.rows]
        height = 26 + sum(row_heights) + 12 * (len(layer.rows) - 1) + 10
        parts.append(f'<rect x="10" y="{y}" width="1080" height="{height}" rx="8" fill="{tone.band}" opacity="0.95"/>')
        parts.append(f'<text x="18" y="{y + 16}" font-size="10" font-weight="700" fill="{tone.head}" '
                     f'letter-spacing="1.5">{escape(layer.name)}</text>')
        row_y = y + 26
        for row_idx, (row, row_h) in enumerate(zip(layer.rows, row_heights)):
            n = len(row)
            width = (1080 - 2 * PAD_X - GAP * (n - 1)) / n
            for i, box in enumerate(row):
                x = 10 + PAD_X + i * (width + GAP)
                _check_fit(box, width, layer.name)
                fill = tone.box if not box.emphasis else tone.box
                stroke_w = "1.6" if box.emphasis else "0.8"
                parts.append(f'<rect x="{x:.1f}" y="{row_y}" width="{width:.1f}" height="{row_h}" rx="6" '
                             f'fill="{fill}" stroke="{tone.stroke}" stroke-width="{stroke_w}"/>')
                cx = x + width / 2
                parts.append(f'<text x="{cx:.1f}" y="{row_y + 22}" text-anchor="middle" font-size="{TITLE_SIZE}" '
                             f'font-weight="600" fill="{tone.title}">{escape(box.title)}</text>')
                for k, line in enumerate(box.lines):
                    parts.append(f'<text x="{cx:.1f}" y="{row_y + 40 + 14 * k}" text-anchor="middle" '
                                 f'font-size="{SUB_SIZE}" fill="{tone.sub}">{escape(line)}</text>')
                if i < n - 1 and row_idx not in layer.support_rows:
                    ax = x + width + 2
                    parts.append(f'<line x1="{ax:.1f}" y1="{row_y + row_h / 2:.1f}" x2="{ax + GAP - 4:.1f}" '
                                 f'y2="{row_y + row_h / 2:.1f}" stroke="{tone.arrow}" stroke-width="1.2" '
                                 f'marker-end="url(#a{li})"/>')
            row_y += row_h + 12
        layer_geometry.append((y, height))
        y += height + 22

    # arrows between layers
    markers.append('<marker id="ag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
                   'orient="auto"><path d="M2 2L8 5L2 8" fill="none" stroke="#6b7280" stroke-width="1.5"/></marker>')
    for (top, h), (nxt, _) in zip(layer_geometry, layer_geometry[1:]):
        parts.append(f'<line x1="550" y1="{top + h + 2}" x2="550" y2="{nxt - 3}" stroke="#6b7280" '
                     f'stroke-width="1.2" marker-end="url(#ag)"/>')

    # footer chips
    chip_y, x = y - 22 + 14, 14
    for text, color, stroke in FOOTER_CHIPS:
        w = _text_width(text, "sub") + 26
        parts.append(f'<rect x="{x:.1f}" y="{chip_y}" width="{w:.1f}" height="26" rx="6" fill="#141a33" '
                     f'stroke="{stroke}" stroke-width="0.8"/>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{chip_y + 17}" text-anchor="middle" font-size="{SUB_SIZE}" '
                     f'fill="{color}">{escape(text)}</text>')
        x += w + 12
    total_height = chip_y + 26 + 16

    head = (f'<svg width="100%" viewBox="0 0 {WIDTH} {total_height}" role="img" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="\'Segoe UI\', Arial, sans-serif">\n'
            f'<title>MathVision AI — System Architecture</title>\n'
            f'<desc>Six-layer architecture: training and data, image preprocessing and handwriting OCR, '
            f'math solver, teaching engine, serving and frontend, quality and deployment.</desc>\n'
            f'<defs>\n  {chr(10).join("  " + m for m in markers).strip()}\n</defs>\n'
            f'<rect width="{WIDTH}" height="{total_height}" fill="#ffffff"/>\n'
            f'<text x="550" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#1a1a2e">'
            f'MathVision AI — System Architecture</text>\n')
    return head + "\n".join(parts) + "\n</svg>\n"


def main() -> None:
    """Write the SVG, and a PNG when cairosvg is installed."""
    svg = build_svg()
    svg_path = OUT_DIR / "architecture.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"wrote {svg_path} ({len(svg) // 1024} KB)")
    try:
        import cairosvg

        png_path = OUT_DIR / "architecture.png"
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=2200,
                         background_color="white")
        print(f"wrote {png_path}")
    except ImportError:
        print("cairosvg not installed: skipped the PNG (pip install cairosvg)")


if __name__ == "__main__":
    main()
