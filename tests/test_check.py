"""Tests for the check module — skill, difficulty, and opposed checks."""

import random

import pytest
from trpg_agent.check import difficulty_check, opposed_check, skill_check


class TestSkillCheck:
    """d100 skill check tests."""

    def test_skill_100_always_success(self):
        """skill_value=100 should always succeed (d100 max is 100)."""
        for _ in range(100):
            result = skill_check(100, 0)
            assert result["success"] is True
            assert result["detail"] == "success"

    def test_skill_0_always_failure(self):
        """skill_value=0 should always fail (d100 min is 1 > 0)."""
        for _ in range(100):
            result = skill_check(0, 0)
            assert result["success"] is False
            assert result["detail"] == "failure"

    def test_positive_modifier_increases_success(self):
        """Positive modifier should increase effective skill value."""
        # seed 9: d100 rolls 60, which is > 50 (no-mod fails) but <= 70 (mod passes)
        random.seed(9)
        result_no_mod = skill_check(50, 0)

        random.seed(9)
        result_with_mod = skill_check(50, 20)

        assert result_no_mod["roll"] == result_with_mod["roll"]
        assert result_no_mod["success"] is False
        assert result_with_mod["success"] is True

    def test_negative_modifier_decreases_success(self):
        """Negative modifier should decrease effective skill value."""
        # seed 3: d100 rolls 31, which is > 30 (neg-mod fails) but <= 50 (no-mod passes)
        random.seed(3)
        result_no_mod = skill_check(50, 0)

        random.seed(3)
        result_neg_mod = skill_check(50, -20)

        assert result_no_mod["roll"] == result_neg_mod["roll"]
        assert result_no_mod["success"] is True
        assert result_neg_mod["success"] is False

    def test_return_keys(self):
        """Result dict should contain required keys."""
        result = skill_check(50, 0)
        assert "success" in result
        assert "detail" in result
        assert "roll" in result
        assert "effective_skill" in result

    def test_modifier_pushed_below_zero(self):
        """Negative modifier can push effective skill below 0."""
        random.seed(1)
        result = skill_check(10, -20)
        assert result["effective_skill"] == -10
        # Any d100 roll will fail against -10
        assert result["success"] is False

    def test_effective_skill_capped_at_max(self):
        """Effective skill can go above 100 with positive modifier."""
        result = skill_check(90, 20)
        assert result["effective_skill"] == 110


class TestDifficultyCheck:
    """d20 difficulty check tests."""

    def test_dc_0_always_success(self):
        """DC=0 should always succeed (d20 min is 1 >= 0)."""
        for _ in range(100):
            result = difficulty_check(0, 0)
            assert result["success"] is True
            assert result["detail"] == "success"

    def test_dc_30_always_failure(self):
        """DC=30 should always fail with no modifier (d20 max is 20 < 30)."""
        for _ in range(100):
            result = difficulty_check(30, 0)
            assert result["success"] is False
            assert result["detail"] == "failure"

    def test_positive_modifier_helps(self):
        """Positive modifier should add to roll."""
        random.seed(7)
        result_no_mod = difficulty_check(15, 0)

        random.seed(7)
        result_with_mod = difficulty_check(15, 5)

        assert result_no_mod["roll"] == result_with_mod["roll"]
        if result_no_mod["success"] is True:
            assert result_with_mod["success"] is True
        else:
            assert result_with_mod["success"] is True

    def test_negative_modifier_hurts(self):
        """Negative modifier should subtract from roll."""
        random.seed(8)
        result_no_mod = difficulty_check(10, 0)

        random.seed(8)
        result_neg_mod = difficulty_check(10, -5)

        assert result_no_mod["roll"] == result_neg_mod["roll"]
        if result_no_mod["success"] is False:
            assert result_neg_mod["success"] is False
        else:
            assert result_neg_mod["success"] is False

    def test_return_keys(self):
        """Result dict should contain required keys."""
        result = difficulty_check(15, 2)
        assert "success" in result
        assert "detail" in result
        assert "roll" in result
        assert "total" in result

    def test_boundary_dc_equals_roll_possible(self):
        """DC=20 with modifier=0 can succeed when d20=20."""
        random.seed(42)
        result = difficulty_check(20, 0)
        # seed 42: d20 rolls should be predictable
        if result["roll"] == 20:
            assert result["success"] is True
        else:
            assert result["success"] is False


class TestOpposedCheck:
    """Opposed d20 check tests."""

    def test_tie_detection(self):
        """When rolls are equal, tie=True and success=False."""
        for _ in range(200):
            result = opposed_check(0, 0)
            if result["roll_player"] == result["roll_opponent"]:
                assert result["success"] is False
                assert result["tie"] is True

    def test_higher_modifier_wins(self):
        """Player with higher modifier should win more often."""
        wins = 0
        for _ in range(100):
            result = opposed_check(10, 0)
            if result["success"]:
                wins += 1
        # Player has a significant advantage, should win > 60%
        assert wins > 60

    def test_return_keys(self):
        """Result dict should contain required keys."""
        result = opposed_check(5, 3)
        assert "success" in result
        assert "detail" in result
        assert "roll_player" in result
        assert "roll_opponent" in result
        assert "total_player" in result
        assert "total_opponent" in result
        assert "tie" in result

    def test_player_win(self):
        """When player total > opponent total, success=True."""
        # Force situation by checking many rolls
        for _ in range(200):
            result = opposed_check(0, 0)
            if result["total_player"] > result["total_opponent"]:
                assert result["success"] is True
                assert result["tie"] is False

    def test_opponent_win(self):
        """When player total < opponent total, success=False, tie=False."""
        for _ in range(200):
            result = opposed_check(0, 0)
            if result["total_player"] < result["total_opponent"]:
                assert result["success"] is False
                assert result["tie"] is False
