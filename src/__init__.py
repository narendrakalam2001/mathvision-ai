"""MathVision AI - core source package.

MathVision AI is an end-to-end handwritten math problem solver and
learning assistant built for Indian school and JEE/NEET students.

Modules exposed by this package (built incrementally):
    config              - central configuration (env vars, paths, constants)
    image_processor      - image preprocessing & quality checks
    ocr_engine            - handwriting -> LaTeX OCR pipeline (TrOCR/Pix2Tex)
    latex_parser          - LaTeX -> SymPy expression conversion
    problem_classifier    - math problem-type detection
    math_solver           - (Day 2) SymPy multi-engine solver
    step_extractor        - (Day 2) human-readable step generation
    llm_explainer         - (Day 2) Gemini-based explanations
    voice_engine          - (Day 2) TTS/STT
    rag_engine            - (Day 2) ChromaDB RAG
    mathbot               - (Day 2) doubt-clearing chatbot
    problem_generator     - (Day 2) practice problem generation
    student_profile       - (Day 2) weak-topic tracking
    graph_plotter         - (Day 2) Matplotlib visualisation
"""

__version__ = "0.1.0"
