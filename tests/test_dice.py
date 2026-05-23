"""Tests for the dice module — dice expression parsing and rolling."""

import pytest
from trpg_agent.dice import parse_dice, roll


class TestParseDice:
    """Tests for parse_dice expression parsing."""

    def test_d20(self):
        """d20 should parse to (1, 20, 0)."""
        assert parse_dice("d20") == (1, 20, 0)

    def test_d100(self):
        """d100 should parse to (1, 100, 0)."""
        assert parse_dice("d100") == (1, 100, 0)

    def test_3d6(self):
        """3d6 should parse to (3, 6, 0)."""
        assert parse_dice("3d6") == (3, 6, 0)

    def test_2d6_plus_3(self):
        """2d6+3 should parse to (2, 6, 3)."""
        assert parse_dice("2d6+3") == (2, 6, 3)

    def test_d20_plus_5(self):
        """d20+5 should parse to (1, 20, 5)."""
        assert parse_dice("d20+5") == (1, 20, 5)

    def test_with_spaces_around_plus(self):
        """2d6 + 3 (with spaces) should parse to (2, 6, 3)."""
        assert parse_dice("2d6 + 3") == (2, 6, 3)

    def test_with_extra_spaces(self):
        """Expression with extra leading/trailing spaces should still parse."""
        assert parse_dice("  3d6  ") == (3, 6, 0)

    def test_empty_expression_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("")

    def test_invalid_string_raises(self):
        """Non-dice string should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("abc")

    def test_no_sides_raises(self):
        """Expression like '2d' with no sides should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("2d")

    def test_just_d_raises(self):
        """Just 'd' should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("d")

    def test_no_dice_part_raises(self):
        """Expression like '+3' without dice part should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("+3")

    def test_negative_modifier_raises(self):
        """Negative modifier like '2d6-3' is not supported and should raise ValueError."""
        with pytest.raises(ValueError):
            parse_dice("2d6-3")


class TestRoll:
    """Tests for dice rolling."""

    def test_d20_returns_single_result(self):
        """d20 should return a list with one element."""
        results, total = roll("d20")
        assert len(results) == 1

    def test_3d6_returns_three_results(self):
        """3d6 should return a list with three elements."""
        results, total = roll("3d6")
        assert len(results) == 3

    def test_d20_results_in_range(self):
        """Each d20 result should be between 1 and 20 (inclusive)."""
        for _ in range(200):
            results, total = roll("d20")
            for r in results:
                assert 1 <= r <= 20

    def test_d100_results_in_range(self):
        """Each d100 result should be between 1 and 100 (inclusive)."""
        for _ in range(100):
            results, total = roll("d100")
            for r in results:
                assert 1 <= r <= 100

    def test_3d6_results_in_range(self):
        """Each 3d6 result should be between 1 and 6 (inclusive)."""
        for _ in range(100):
            results, total = roll("3d6")
            for r in results:
                assert 1 <= r <= 6

    def test_total_without_modifier(self):
        """Total should equal sum of individual results when no modifier."""
        for _ in range(50):
            results, total = roll("3d6")
            assert total == sum(results)

    def test_total_with_modifier(self):
        """Total should equal sum of individual results plus modifier."""
        for _ in range(50):
            results, total = roll("2d6+3")
            assert total == sum(results) + 3

    def test_total_with_modifier_and_spaces(self):
        """Total should work correctly with spaced expression."""
        for _ in range(50):
            results, total = roll("2d6 + 3")
            assert total == sum(results) + 3

    def test_roll_invalid_expr_raises(self):
        """Invalid expression should raise ValueError from roll()."""
        with pytest.raises(ValueError):
            roll("xyz")

    def test_roll_empty_raises(self):
        """Empty expression should raise ValueError from roll()."""
        with pytest.raises(ValueError):
            roll("")

    def test_deterministic_seed(self):
        """With fixed random seed, roll should produce predictable output.
        This test verifies that roll() uses random in a standard way."""
        import random
        random.seed(42)
        results1, total1 = roll("3d6")

        random.seed(42)
        results2, total2 = roll("3d6")

        assert results1 == results2
        assert total1 == total2
