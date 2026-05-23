"""Lightweight rule-driven state machine for NPC emotion/trust/stamina.

Tracks three dimensions:
  - emotion:  calm -> wary -> hostile  (ordered, 3 levels)
  - trust:    0.0 ~ 1.0               (continuous float)
  - stamina:  fresh -> tired -> exhausted  (ordered, 3 levels)

Trigger rules modify these dimensions; unknown triggers are silently ignored.
All dimensions are clamped at their boundaries.
"""

from __future__ import annotations

import copy
from typing import Any

# Ordered emotion levels (index 0 = calm, 1 = wary, 2 = hostile)
_EMOTION_LEVELS = ["calm", "wary", "hostile"]

# Ordered stamina levels (index 0 = fresh, 1 = tired, 2 = exhausted)
_STAMINA_LEVELS = ["fresh", "tired", "exhausted"]

# Trigger -> list of (dimension, delta) effects
# delta for ordered dimensions is an integer index shift;
# delta for trust is a float addend.
_TRIGGER_RULES: dict[str, list[tuple[str, Any]]] = {
    "betrayed":   [("emotion", 1), ("trust", -0.1)],
    "helped":     [("emotion", -1), ("trust", 0.1)],
    "combat":     [("stamina", 1)],
    "rested":     [("stamina", -1)],
    "threatened": [("emotion", 1)],
    "gifted":     [("trust", 0.1)],
}


class StateMachine:
    """A lightweight, rule-driven state machine for NPC affect and condition.

    Parameters
    ----------
    emotion : str, optional
        Initial emotion level (default ``"calm"``).
    trust : float, optional
        Initial trust value between 0.0 and 1.0 (default ``0.5``).
    stamina : str, optional
        Initial stamina level (default ``"fresh"``).
    """

    def __init__(
        self,
        emotion: str = "calm",
        trust: float = 0.5,
        stamina: str = "fresh",
    ) -> None:
        # Validate initial values
        if emotion not in _EMOTION_LEVELS:
            raise ValueError(f"Invalid emotion: {emotion!r}. Must be one of {_EMOTION_LEVELS}")
        if not 0.0 <= trust <= 1.0:
            raise ValueError(f"Trust must be between 0.0 and 1.0, got {trust}")
        if stamina not in _STAMINA_LEVELS:
            raise ValueError(f"Invalid stamina: {stamina!r}. Must be one of {_STAMINA_LEVELS}")

        self._emotion_idx = _EMOTION_LEVELS.index(emotion)
        self._trust = trust
        self._stamina_idx = _STAMINA_LEVELS.index(stamina)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the current state.

        Returns
        -------
        dict
            A dictionary with keys ``emotion``, ``trust``, and ``stamina``.
        """
        return {
            "emotion": _EMOTION_LEVELS[self._emotion_idx],
            "trust": round(self._trust, 2),
            "stamina": _STAMINA_LEVELS[self._stamina_idx],
        }

    def apply(self, trigger: str) -> None:
        """Apply a trigger to update the state machine.

        Parameters
        ----------
        trigger : str
            The trigger name.  Unknown triggers are silently ignored.
        """
        effects = _TRIGGER_RULES.get(trigger)
        if effects is None:
            return None

        for dimension, delta in effects:
            if dimension == "emotion":
                new_idx = self._emotion_idx + delta
                self._emotion_idx = max(0, min(len(_EMOTION_LEVELS) - 1, new_idx))
            elif dimension == "trust":
                self._trust = max(0.0, min(1.0, self._trust + delta))
            elif dimension == "stamina":
                new_idx = self._stamina_idx + delta
                self._stamina_idx = max(0, min(len(_STAMINA_LEVELS) - 1, new_idx))

        return None
