"""Tests for curriculum learning manager."""

import pytest

from src.training.curriculum import DEFAULT_STAGES, CurriculumManager


class TestCurriculumManagerInit:
    """Tests for CurriculumManager initialization and default state."""

    def test_curriculum_initial_stage(self):
        """Default manager starts at 'short' stage with max_steps 10."""
        manager = CurriculumManager()
        assert manager.current_stage["name"] == "short"
        assert manager.current_max_steps == 10

    def test_curriculum_default_stages(self):
        """DEFAULT_STAGES has 4 entries with expected names."""
        names = [s["name"] for s in DEFAULT_STAGES]
        assert names == ["short", "medium", "long", "full"]

    def test_curriculum_custom_stages(self):
        """Manager accepts custom stages."""
        custom = [
            {"name": "easy", "max_steps": 5, "advance_threshold": 0.8},
            {"name": "hard", "max_steps": 10, "advance_threshold": 0.0},
        ]
        manager = CurriculumManager(stages=custom)
        assert manager.current_stage["name"] == "easy"
        assert manager.current_max_steps == 5

    def test_get_config_returns_dict(self):
        """get_config() returns a dict with stage info."""
        manager = CurriculumManager()
        config = manager.get_config()
        assert isinstance(config, dict)
        assert "stage_name" in config
        assert "max_steps" in config
        assert "advance_threshold" in config
        assert config["stage_name"] == "short"
        assert config["max_steps"] == 10


class TestCurriculumAdvance:
    """Tests for performance recording and stage advancement."""

    def test_curriculum_advance(self):
        """Recording 3 performances >= 0.7 should advance from short to medium."""
        manager = CurriculumManager()
        for _ in range(3):
            manager.record_performance(0.7)
        assert manager.should_advance() is True
        manager.advance()
        assert manager.current_stage["name"] == "medium"
        assert manager.current_max_steps == 20

    def test_should_not_advance_below_threshold(self):
        """Performances below threshold should not trigger advancement."""
        manager = CurriculumManager()
        for _ in range(3):
            manager.record_performance(0.5)
        assert manager.should_advance() is False

    def test_advance_resets_performance_window(self):
        """After advancing, performance history is reset."""
        manager = CurriculumManager()
        for _ in range(3):
            manager.record_performance(0.7)
        manager.advance()
        # After advancing, we should need new performances to advance again
        assert manager.should_advance() is False

    def test_window_size_limits_history(self):
        """Only the last window_size performances are considered."""
        manager = CurriculumManager(window_size=5)
        # Record 5 bad performances
        for _ in range(5):
            manager.record_performance(0.3)
        # Record 3 good performances (within window of 5, so old ones drop)
        for _ in range(3):
            manager.record_performance(0.9)
        # Window of 5: [0.3, 0.3, 0.9, 0.9, 0.9] -> avg = 0.66 < 0.7
        assert manager.should_advance() is False

    def test_advance_through_all_stages(self):
        """Can advance through all stages to the end."""
        manager = CurriculumManager()
        stages_seen = []
        while True:
            stages_seen.append(manager.current_stage["name"])
            threshold = manager.current_stage["advance_threshold"]
            # Last stage has threshold 0.0, should_advance returns False
            if threshold == 0.0:
                break
            for _ in range(3):
                manager.record_performance(threshold)
            assert manager.should_advance() is True
            manager.advance()
        assert stages_seen == ["short", "medium", "long", "full"]

    def test_advance_at_final_stage_is_noop(self):
        """Advancing past the last stage does nothing."""
        manager = CurriculumManager()
        # Fast-forward to last stage
        while manager._stage_index < len(manager._stages) - 1:
            threshold = manager.current_stage["advance_threshold"]
            for _ in range(3):
                manager.record_performance(threshold)
            manager.advance()
        last_name = manager.current_stage["name"]
        manager.advance()  # should be a no-op
        assert manager.current_stage["name"] == last_name
