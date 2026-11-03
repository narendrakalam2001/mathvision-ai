"""Student profile & weak-topic tracker (feeds Screen 5's Dashboard).

Tracks per-student solving history, computes a topic-strength heatmap
("Strong" / "Good" / "Need Practice" / "Weak"), maintains a daily
solving streak, and awards the achievement badges from the product
spec ("Problem Solver", "Week Streak", "Speed Demon", "Perfect Week").

Persistence is a simple on-disk JSON store keyed by ``student_id`` —
no database dependency required for the MVP; the storage backend is
isolated behind :class:`StudentProfileStore` so it can be swapped for
a real database (Postgres/Firestore) later without touching the
tracking logic in :class:`StudentProfile`.

Example:
    >>> from src.student_profile import StudentProfileStore
    >>> store = StudentProfileStore()
    >>> profile = store.get_or_create("student_123")
    >>> profile.record_attempt(topic="Trigonometry", correct=False, solve_time_seconds=95)
    >>> store.save(profile)
    >>> profile.weak_topics()
    ['Trigonometry']
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


class StrengthLevel:
    """String constants for the topic-heatmap strength bands."""

    STRONG = "Strong"
    GOOD = "Good"
    NEED_PRACTICE = "Need Practice"
    WEAK = "Weak"


_STRENGTH_THRESHOLDS: list[tuple[float, str]] = [
    (0.85, StrengthLevel.STRONG),
    (0.65, StrengthLevel.GOOD),
    (0.40, StrengthLevel.NEED_PRACTICE),
    (0.0, StrengthLevel.WEAK),
]


@dataclass
class TopicStats:
    """Running accuracy stats for one topic.

    Attributes:
        attempts: Total problems attempted in this topic.
        correct: Number marked correct (self-assessed by the student
            via the "Got it" / "Doubt?" buttons, or verified by the
            solver when auto-graded).
    """

    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0

    @property
    def strength(self) -> str:
        if self.attempts == 0:
            return StrengthLevel.NEED_PRACTICE
        for threshold, level in _STRENGTH_THRESHOLDS:
            if self.accuracy >= threshold:
                return level
        return StrengthLevel.WEAK


@dataclass
class SolvedRecord:
    """One entry in the "Recent Solutions" list.

    Attributes:
        latex_problem: The problem's LaTeX.
        topic: The classified topic.
        timestamp: ISO-8601 timestamp string.
        solve_time_seconds: Time taken to solve, for the "Speed Demon" badge.
        correct: Self-assessed correctness.
    """

    latex_problem: str
    topic: str
    timestamp: str
    solve_time_seconds: float
    correct: bool


@dataclass
class StudentProfile:
    """A single student's full tracked profile.

    Attributes:
        student_id: Unique identifier.
        name: Display name.
        class_level: e.g. "Class 10".
        topic_stats: Per-topic :class:`TopicStats`.
        recent_solutions: Most recent solved problems (capped list).
        streak_days: Current consecutive-day solving streak.
        last_solved_date: ISO date string of the last day a problem was solved.
        total_solved: Lifetime total problems solved.
        badges: Set of earned badge names.
        notebook: Problems the student bookmarked with "Save to Notebook"
            (kept separately from the automatic recent-solutions list).
    """

    student_id: str
    name: str = "Student"
    class_level: str = "Class 10"
    topic_stats: dict[str, TopicStats] = field(default_factory=dict)
    recent_solutions: list[SolvedRecord] = field(default_factory=list)
    streak_days: int = 0
    last_solved_date: Optional[str] = None
    total_solved: int = 0
    badges: list[str] = field(default_factory=list)
    notebook: list[dict] = field(default_factory=list)

    MAX_RECENT: int = field(default=20, repr=False, compare=False)
    MAX_NOTEBOOK: int = field(default=100, repr=False, compare=False)

    # ------------------------------------------------------------------
    def record_attempt(
        self,
        topic: str,
        correct: bool,
        solve_time_seconds: float,
        latex_problem: str = "",
    ) -> list[str]:
        """Record a solved (or attempted) problem and update all stats.

        Args:
            topic: The problem's classified topic (e.g. "Quadratic Equations").
            correct: Whether the student marked it as correctly understood/solved.
            solve_time_seconds: How long the attempt took.
            latex_problem: The problem's LaTeX, for the recent-solutions list.

        Returns:
            List of newly-earned badge names (empty if none this call).
        """
        stats = self.topic_stats.setdefault(topic, TopicStats())
        stats.attempts += 1
        if correct:
            stats.correct += 1

        record = SolvedRecord(
            latex_problem=latex_problem,
            topic=topic,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            solve_time_seconds=solve_time_seconds,
            correct=correct,
        )
        self.recent_solutions.insert(0, record)
        self.recent_solutions = self.recent_solutions[: self.MAX_RECENT]

        self.total_solved += 1
        self._update_streak()

        return self._check_new_badges(solve_time_seconds)

    def mark_last_attempt(self, correct: bool) -> bool:
        """Change the self-assessment of the most recent attempt.

        Problems are recorded as understood when the solution is shown; if the
        student then says "I didn't understand", this flips that record so the
        topic accuracy, weak-topic list and heatmap reflect it.

        Args:
            correct: New self-assessed value for the latest attempt.

        Returns:
            ``True`` if something changed, ``False`` if there is no attempt or
            it already had that value.
        """
        if not self.recent_solutions or self.recent_solutions[0].correct == correct:
            return False
        latest = self.recent_solutions[0]
        latest.correct = correct
        stats = self.topic_stats.get(latest.topic)
        if stats is not None:
            stats.correct = max(0, min(stats.attempts, stats.correct + (1 if correct else -1)))
        return True

    def save_to_notebook(self, latex_problem: str, topic: str, answer: str = "") -> bool:
        """Bookmark a problem (no effect on the solved/accuracy statistics).

        Args:
            latex_problem: Problem LaTeX.
            topic: Topic label.
            answer: Final answer text.

        Returns:
            ``True`` if added, ``False`` if the same problem was already saved.
        """
        if any(entry["latex_problem"] == latex_problem for entry in self.notebook):
            return False
        self.notebook.insert(0, {
            "latex_problem": latex_problem, "topic": topic, "answer": answer,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.notebook = self.notebook[: self.MAX_NOTEBOOK]
        return True

    def _update_streak(self) -> None:
        """Increment/reset the daily streak based on ``last_solved_date``."""
        today = date.today()
        if self.last_solved_date is None:
            self.streak_days = 1
        else:
            last = date.fromisoformat(self.last_solved_date)
            delta = (today - last).days
            if delta == 0:
                pass  # already solved something today, streak unchanged
            elif delta == 1:
                self.streak_days += 1
            else:
                self.streak_days = 1
        self.last_solved_date = today.isoformat()

    def _check_new_badges(self, solve_time_seconds: float) -> list[str]:
        """Check badge-earning conditions and append any newly-earned ones."""
        newly_earned = []

        def award(name: str) -> None:
            if name not in self.badges:
                self.badges.append(name)
                newly_earned.append(name)

        if self.total_solved >= 100:
            award("Problem Solver")
        if self.streak_days >= 7:
            award("Week Streak")
        if solve_time_seconds < 30:
            award("Speed Demon")
        if self._is_perfect_week():
            award("Perfect Week")

        return newly_earned

    def _is_perfect_week(self) -> bool:
        """True if every attempt in the last 7 days was marked correct."""
        cutoff = datetime.now() - timedelta(days=7)
        recent = [
            r for r in self.recent_solutions if datetime.fromisoformat(r.timestamp) >= cutoff
        ]
        return bool(recent) and all(r.correct for r in recent)

    # ------------------------------------------------------------------
    def weak_topics(self) -> list[str]:
        """Return topics currently at 'Weak' or 'Need Practice' strength."""
        return [
            topic
            for topic, stats in self.topic_stats.items()
            if stats.strength in (StrengthLevel.WEAK, StrengthLevel.NEED_PRACTICE)
        ]

    def heatmap(self) -> dict[str, dict[str, object]]:
        """Return the full topic heatmap for Screen 5's dashboard.

        Returns:
            Mapping of topic -> {"accuracy": float, "strength": str, "attempts": int}.
        """
        return {
            topic: {
                "accuracy": round(stats.accuracy, 3),
                "strength": stats.strength,
                "attempts": stats.attempts,
            }
            for topic, stats in self.topic_stats.items()
        }

    def dashboard_summary(self) -> dict[str, object]:
        """Return the aggregate stats block for the top of Screen 5.

        Returns:
            A dict with total_solved, streak_days, average_solve_time,
            and self-assessed accuracy — matching the product spec's
            "Performance Overview" card.
        """
        today_str = date.today().isoformat()
        solved_today = sum(1 for r in self.recent_solutions if r.timestamp.startswith(today_str))
        week_cutoff = datetime.now() - timedelta(days=7)
        solved_this_week = sum(
            1 for r in self.recent_solutions if datetime.fromisoformat(r.timestamp) >= week_cutoff
        )
        times = [r.solve_time_seconds for r in self.recent_solutions] or [0.0]
        correct_count = sum(1 for r in self.recent_solutions if r.correct)
        accuracy = correct_count / len(self.recent_solutions) if self.recent_solutions else 0.0

        return {
            "total_solved": self.total_solved,
            "solved_today": solved_today,
            "solved_this_week": solved_this_week,
            "average_solve_time_seconds": round(sum(times) / len(times), 1),
            "accuracy_pct": round(accuracy * 100, 1),
            "streak_days": self.streak_days,
            "badges": list(self.badges),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("MAX_RECENT", None)
        data.pop("MAX_NOTEBOOK", None)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        topic_stats = {k: TopicStats(**v) for k, v in data.get("topic_stats", {}).items()}
        recent_solutions = [SolvedRecord(**r) for r in data.get("recent_solutions", [])]
        return cls(
            student_id=data["student_id"],
            name=data.get("name", "Student"),
            class_level=data.get("class_level", "Class 10"),
            topic_stats=topic_stats,
            recent_solutions=recent_solutions,
            streak_days=data.get("streak_days", 0),
            last_solved_date=data.get("last_solved_date"),
            total_solved=data.get("total_solved", 0),
            badges=data.get("badges", []),
            notebook=list(data.get("notebook", [])),
        )


class StudentProfileStore:
    """JSON-file-backed persistence for :class:`StudentProfile` objects."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        """Create a store.

        Args:
            storage_dir: Directory to hold one JSON file per student.
                Defaults to ``settings.student_profile_dir`` (env ``STUDENT_PROFILE_DIR``).
        """
        self.storage_dir = storage_dir or settings.student_profile_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, student_id: str) -> Path:
        safe_id = "".join(c for c in student_id if c.isalnum() or c in ("-", "_")) or "unknown"
        return self.storage_dir / f"{safe_id}.json"

    def get_or_create(self, student_id: str, name: str = "Student", class_level: str = "Class 10") -> StudentProfile:
        """Load an existing profile or create a fresh one.

        Args:
            student_id: Unique student identifier.
            name: Display name, used only when creating a new profile.
            class_level: Class level, used only when creating a new profile.

        Returns:
            The loaded or newly-created :class:`StudentProfile`.
        """
        path = self._path_for(student_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return StudentProfile.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError):
                logger.warning("Corrupt profile for %s, creating fresh one", student_id, exc_info=True)

        return StudentProfile(student_id=student_id, name=name, class_level=class_level)

    def save(self, profile: StudentProfile) -> None:
        """Persist a profile to disk.

        Args:
            profile: The profile to save.
        """
        path = self._path_for(profile.student_id)
        try:
            path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            logger.error("Failed to save profile for %s", profile.student_id, exc_info=True)


if __name__ == "__main__":
    store = StudentProfileStore()
    profile = store.get_or_create("demo_student", name="Narendra", class_level="Class 12")

    profile.record_attempt("Quadratic Equations", correct=True, solve_time_seconds=45, latex_problem="x^2-5x+6=0")
    profile.record_attempt("Trigonometry", correct=False, solve_time_seconds=120)
    profile.record_attempt("Trigonometry", correct=False, solve_time_seconds=90)
    badges = profile.record_attempt("Calculus", correct=True, solve_time_seconds=20)

    print("Heatmap:", profile.heatmap())
    print("Weak topics:", profile.weak_topics())
    print("Dashboard summary:", profile.dashboard_summary())
    print("Newly earned badges:", badges)

    store.save(profile)
    print(f"Saved profile to {store._path_for('demo_student')}")
