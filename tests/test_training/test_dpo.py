"""Tests for DPOTrainer and PreferencePair."""

import pytest

from src.training.dpo_trainer import DPOTrainer, PreferencePair


class TestDPOTrainerInit:
    """Tests for DPOTrainer initialization."""

    def test_dpo_trainer_init(self):
        """DPOTrainer(max_length=32768) has max_length==32768."""
        trainer = DPOTrainer(max_length=32768)
        assert trainer.max_length == 32768

    def test_dpo_trainer_default_beta(self):
        """DPOTrainer has default beta=0.1."""
        trainer = DPOTrainer()
        assert trainer.beta == 0.1

    def test_dpo_trainer_custom_beta(self):
        """DPOTrainer accepts custom beta."""
        trainer = DPOTrainer(beta=0.2)
        assert trainer.beta == 0.2

    def test_dpo_trainer_default_max_length(self):
        """DPOTrainer has default max_length=32768."""
        trainer = DPOTrainer()
        assert trainer.max_length == 32768


class TestBuildPreferencePair:
    """Tests for DPOTrainer.build_preference_pair."""

    def test_dpo_build_preference_pair(self):
        """build_preference_pair(chosen, rejected) returns dict with chosen and rejected keys."""
        trainer = DPOTrainer()
        result = trainer.build_preference_pair(
            chosen="好的旅行方案",
            rejected="不好的旅行方案",
        )
        assert isinstance(result, dict)
        assert "chosen" in result
        assert "rejected" in result

    def test_dpo_build_preference_pair_values(self):
        """build_preference_pair stores chosen and rejected values correctly."""
        trainer = DPOTrainer()
        result = trainer.build_preference_pair(
            chosen="方案A",
            rejected="方案B",
        )
        assert result["chosen"] == "方案A"
        assert result["rejected"] == "方案B"

    def test_dpo_build_preference_pair_with_query(self):
        """build_preference_pair stores optional query."""
        trainer = DPOTrainer()
        result = trainer.build_preference_pair(
            chosen="方案A",
            rejected="方案B",
            query="北京三日游",
        )
        assert result["query"] == "北京三日游"

    def test_dpo_build_preference_pair_with_constraints(self):
        """build_preference_pair stores optional constraints."""
        trainer = DPOTrainer()
        constraints = [{"type": "budget", "value": "5000"}]
        result = trainer.build_preference_pair(
            chosen="方案A",
            rejected="方案B",
            constraints=constraints,
        )
        assert result["constraints"] == constraints

    def test_dpo_build_preference_pair_default_query(self):
        """build_preference_pair defaults query to empty string."""
        trainer = DPOTrainer()
        result = trainer.build_preference_pair(
            chosen="方案A",
            rejected="方案B",
        )
        assert result["query"] == ""

    def test_dpo_build_preference_pair_default_constraints(self):
        """build_preference_pair defaults constraints to None."""
        trainer = DPOTrainer()
        result = trainer.build_preference_pair(
            chosen="方案A",
            rejected="方案B",
        )
        assert result["constraints"] is None


class TestPreferencePairDataclass:
    """Tests for PreferencePair dataclass."""

    def test_preference_pair_creation(self):
        """PreferencePair can be created with chosen and rejected."""
        pair = PreferencePair(chosen="好方案", rejected="差方案")
        assert pair.chosen == "好方案"
        assert pair.rejected == "差方案"

    def test_preference_pair_defaults(self):
        """PreferencePair has default empty query and None constraints."""
        pair = PreferencePair(chosen="A", rejected="B")
        assert pair.query == ""
        assert pair.constraints is None

    def test_preference_pair_with_all_fields(self):
        """PreferencePair stores all fields correctly."""
        constraints = [{"type": "budget", "value": "3000"}]
        pair = PreferencePair(
            chosen="好",
            rejected="差",
            query="测试",
            constraints=constraints,
        )
        assert pair.query == "测试"
        assert pair.constraints == constraints


class TestGeneratePreferenceData:
    """Tests for DPOTrainer.generate_preference_data."""

    def test_generate_preference_data_returns_list(self):
        """generate_preference_data returns a list."""
        trainer = DPOTrainer()
        result = trainer.generate_preference_data([])
        assert isinstance(result, list)

    def test_generate_preference_data_empty_episodes(self):
        """generate_preference_data returns empty list for empty episodes."""
        trainer = DPOTrainer()
        result = trainer.generate_preference_data([])
        assert len(result) == 0
