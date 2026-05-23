"""Three check types for TRPG: skill check, difficulty check, opposed check.

All random numbers are generated through :func:`dice.roll` — never
call ``random`` directly in this module.
"""

from __future__ import annotations

from trpg_agent.dice import roll


def skill_check(skill_value: int, modifier: int = 0) -> dict:
    """d100 skill check.

    Success when ``d100_roll <= skill_value + modifier``.

    Parameters
    ----------
    skill_value : int
        Base skill value (0-100 typical).
    modifier : int
        Modifier applied to *skill_value* (positive = easier).

    Returns
    -------
    dict
        ``{"success": bool, "detail": str, "roll": int, "effective_skill": int}``
    """
    _, total = roll("d100")
    effective_skill = skill_value + modifier
    success = total <= effective_skill
    detail = "success" if success else "failure"
    return {
        "success": success,
        "detail": detail,
        "roll": total,
        "effective_skill": effective_skill,
    }


def difficulty_check(dc: int, modifier: int = 0) -> dict:
    """d20 difficulty check.

    Success when ``d20_roll + modifier >= DC``.

    Parameters
    ----------
    dc : int
        Difficulty class (target number).
    modifier : int
        Modifier added to the d20 roll (positive = helps).

    Returns
    -------
    dict
        ``{"success": bool, "detail": str, "roll": int, "total": int}``
    """
    _, roll_total = roll("d20")
    total = roll_total + modifier
    success = total >= dc
    detail = "success" if success else "failure"
    return {
        "success": success,
        "detail": detail,
        "roll": roll_total,
        "total": total,
    }


def opposed_check(player_mod: int, opponent_mod: int) -> dict:
    """Opposed d20 check.

    Both sides roll d20 + their modifier; higher total wins.
    A tie is recorded as a failure for the player with ``tie=True``.

    Parameters
    ----------
    player_mod : int
        Modifier for the player (positive = helps).
    opponent_mod : int
        Modifier for the opponent.

    Returns
    -------
    dict
        ``{"success": bool, "detail": str, "roll_player": int,
        "roll_opponent": int, "total_player": int,
        "total_opponent": int, "tie": bool}``
    """
    _, player_roll = roll("d20")
    _, opponent_roll = roll("d20")
    player_total = player_roll + player_mod
    opponent_total = opponent_roll + opponent_mod

    if player_total > opponent_total:
        success = True
        tie = False
        detail = "success"
    elif player_total < opponent_total:
        success = False
        tie = False
        detail = "failure"
    else:
        success = False
        tie = True
        detail = "tie"

    return {
        "success": success,
        "detail": detail,
        "roll_player": player_roll,
        "roll_opponent": opponent_roll,
        "total_player": player_total,
        "total_opponent": opponent_total,
        "tie": tie,
    }
