"""Dice expression parser and roller — the single random-number entry point for the project.

All modules that need randomisation (checks, events, etc.) must call :func:`roll`,
never ``random`` directly.

Supported expressions:
    - ``d20``         → 1d20 + 0
    - ``d100``        → 1d100 + 0
    - ``3d6``         → 3d6 + 0
    - ``2d6+3``       → 2d6 + 3
    - ``2d6 + 3``     → spaces around ``+`` are tolerated
"""

from __future__ import annotations

import random
import re
from typing import List, Tuple

# Regex for "ndm+k" with optional spaces around '+'.
# Groups: (count, sides, modifier)
# - count is optional (defaults to 1 if missing)
# - sides is required (one or more digits)
# - modifier is optional (one or more digits after '+')
_EXPR_RE = re.compile(r"^(\d*)d(\d+)(?:\s*\+\s*(\d+))?$")


def parse_dice(expr: str) -> Tuple[int, int, int]:
    """Parse a dice expression and return ``(count, sides, modifier)``.

    Parameters
    ----------
    expr : str
        Dice expression such as ``"d20"``, ``"3d6"``, or ``"2d6+3"``.
        Spaces around ``+`` are allowed.

    Returns
    -------
    tuple of (int, int, int)
        ``(number_of_dice, number_of_sides, flat_modifier)``.

    Raises
    ------
    ValueError
        If the expression is empty, does not match the expected format,
        or has invalid (non-positive) count/sides.
    """
    stripped = expr.strip()
    if not stripped:
        raise ValueError(f"Invalid dice expression: {expr!r}")

    m = _EXPR_RE.match(stripped)
    if m is None:
        raise ValueError(f"Invalid dice expression: {expr!r}")

    count_str, sides_str, mod_str = m.groups()

    count = int(count_str) if count_str else 1
    sides = int(sides_str)
    modifier = int(mod_str) if mod_str else 0

    if count < 1:
        raise ValueError(f"Dice count must be positive, got {count}")
    if sides < 1:
        raise ValueError(f"Number of sides must be positive, got {sides}")

    return count, sides, modifier


def roll(expr: str) -> Tuple[List[int], int]:
    """Roll dice according to *expr* and return ``(individual_results, total)``.

    Parameters
    ----------
    expr : str
        Dice expression (see module docstring for supported formats).

    Returns
    -------
    tuple of (list of int, int)
        A list of each die's result and the grand total (sum of dice
        results plus any flat modifier).

    Raises
    ------
    ValueError
        Propagated from :func:`parse_dice` if *expr* is invalid.
    """
    count, sides, modifier = parse_dice(expr)
    results = [random.randint(1, sides) for _ in range(count)]
    total = sum(results) + modifier
    return results, total
