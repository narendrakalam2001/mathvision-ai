"""Notebook bookmarks and self-assessment changes on StudentProfile / StudentProfileStore."""

from __future__ import annotations

import pytest

from src.student_profile import StudentProfile, StudentProfileStore


@pytest.fixture()
def profile() -> StudentProfile:
    return StudentProfile(student_id="s1")


def _solve(profile: StudentProfile, topic: str = "Quadratic Equation", correct: bool = True) -> None:
    profile.record_attempt(topic=topic, correct=correct, solve_time_seconds=12.0, latex_problem="x^2=4")


class TestMarkLastAttempt:
    def test_flip_to_confused_updates_topic_accuracy(self, profile) -> None:
        _solve(profile)
        assert profile.mark_last_attempt(False) is True
        assert profile.recent_solutions[0].correct is False
        assert profile.topic_stats["Quadratic Equation"].correct == 0
        assert "Quadratic Equation" in profile.weak_topics()

    def test_flip_back_restores_accuracy(self, profile) -> None:
        _solve(profile)
        profile.mark_last_attempt(False)
        assert profile.mark_last_attempt(True) is True
        assert profile.topic_stats["Quadratic Equation"].correct == 1
        assert profile.weak_topics() == []

    def test_same_value_is_a_no_op(self, profile) -> None:
        _solve(profile)
        assert profile.mark_last_attempt(True) is False
        assert profile.topic_stats["Quadratic Equation"].correct == 1

    def test_only_the_latest_attempt_changes(self, profile) -> None:
        _solve(profile, "Linear Equation")
        _solve(profile, "Quadratic Equation")
        profile.mark_last_attempt(False)
        assert profile.topic_stats["Linear Equation"].correct == 1

    def test_no_attempts_is_safe(self, profile) -> None:
        assert profile.mark_last_attempt(False) is False

    def test_counts_never_go_negative(self, profile) -> None:
        _solve(profile)
        profile.mark_last_attempt(False)
        profile.recent_solutions[0].correct = True           # corrupted state
        profile.mark_last_attempt(False)
        assert profile.topic_stats["Quadratic Equation"].correct == 0


class TestNotebook:
    def test_save_and_list(self, profile) -> None:
        assert profile.save_to_notebook("x^2 = 4", "Quadratic Equation", "x = ±2") is True
        entry = profile.notebook[0]
        assert entry["latex_problem"] == "x^2 = 4" and entry["answer"] == "x = ±2" and entry["saved_at"]

    def test_duplicates_are_ignored(self, profile) -> None:
        profile.save_to_notebook("x^2 = 4", "Quadratic Equation")
        assert profile.save_to_notebook("x^2 = 4", "Quadratic Equation") is False
        assert len(profile.notebook) == 1

    def test_newest_first_and_capped(self, profile) -> None:
        for i in range(105):
            profile.save_to_notebook(f"x = {i}", "Linear Equation")
        assert profile.notebook[0]["latex_problem"] == "x = 104" and len(profile.notebook) == 100

    def test_saving_does_not_change_solved_statistics(self, profile) -> None:
        profile.save_to_notebook("x^2 = 4", "Quadratic Equation")
        assert profile.total_solved == 0 and not profile.recent_solutions

    def test_roundtrip_through_the_store(self, tmp_path) -> None:
        store = StudentProfileStore(storage_dir=tmp_path)
        profile = store.get_or_create("s9")
        profile.save_to_notebook("x^2 = 4", "Quadratic Equation", "x = ±2")
        store.save(profile)
        assert store.get_or_create("s9").notebook[0]["answer"] == "x = ±2"

    def test_old_profile_files_without_a_notebook_still_load(self, tmp_path) -> None:
        store = StudentProfileStore(storage_dir=tmp_path)
        profile = store.get_or_create("legacy")
        data = profile.to_dict()
        data.pop("notebook", None)
        assert StudentProfile.from_dict(data).notebook == []
