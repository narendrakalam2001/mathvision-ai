"""Unit tests for src.student_profile."""

from __future__ import annotations

import pytest

from src.student_profile import StrengthLevel, StudentProfile, StudentProfileStore


@pytest.fixture()
def profile() -> StudentProfile:
    return StudentProfile(student_id="test_student")


class TestTopicStats:
    def test_strong_topic_after_all_correct(self, profile: StudentProfile) -> None:
        for _ in range(5):
            profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        assert profile.topic_stats["Algebra"].strength == StrengthLevel.STRONG

    def test_weak_topic_after_all_incorrect(self, profile: StudentProfile) -> None:
        for _ in range(3):
            profile.record_attempt("Calculus", correct=False, solve_time_seconds=60)
        assert profile.topic_stats["Calculus"].strength == StrengthLevel.WEAK
        assert "Calculus" in profile.weak_topics()


class TestStreak:
    def test_first_attempt_sets_streak_to_one(self, profile: StudentProfile) -> None:
        profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        assert profile.streak_days == 1

    def test_same_day_does_not_increment_streak(self, profile: StudentProfile) -> None:
        profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        assert profile.streak_days == 1


class TestBadges:
    def test_speed_demon_badge_awarded_under_30s(self, profile: StudentProfile) -> None:
        newly_earned = profile.record_attempt("Algebra", correct=True, solve_time_seconds=15)
        assert "Speed Demon" in newly_earned
        assert "Speed Demon" in profile.badges

    def test_no_duplicate_badge_awards(self, profile: StudentProfile) -> None:
        profile.record_attempt("Algebra", correct=True, solve_time_seconds=15)
        newly_earned_again = profile.record_attempt("Algebra", correct=True, solve_time_seconds=15)
        assert "Speed Demon" not in newly_earned_again  # already earned, not re-awarded

    def test_problem_solver_badge_at_100(self, profile: StudentProfile) -> None:
        for _ in range(99):
            profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        newly_earned = profile.record_attempt("Algebra", correct=True, solve_time_seconds=60)
        assert "Problem Solver" in newly_earned


class TestDashboardSummary:
    def test_summary_reflects_attempts(self, profile: StudentProfile) -> None:
        profile.record_attempt("Algebra", correct=True, solve_time_seconds=30, latex_problem="2x=4")
        summary = profile.dashboard_summary()
        assert summary["total_solved"] == 1
        assert summary["solved_today"] == 1
        assert summary["accuracy_pct"] == 100.0


class TestStudentProfileStore:
    def test_save_and_reload_roundtrip(self, tmp_path) -> None:
        store = StudentProfileStore(storage_dir=tmp_path)
        profile = store.get_or_create("student_x", name="Priya", class_level="Class 9")
        profile.record_attempt("Trigonometry", correct=False, solve_time_seconds=80)
        store.save(profile)

        reloaded = store.get_or_create("student_x")
        assert reloaded.name == "Priya"
        assert reloaded.total_solved == 1
        assert "Trigonometry" in reloaded.topic_stats

    def test_get_or_create_new_student(self, tmp_path) -> None:
        store = StudentProfileStore(storage_dir=tmp_path)
        profile = store.get_or_create("brand_new_student")
        assert profile.total_solved == 0
        assert profile.badges == []
