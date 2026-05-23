"""Lightweight rule-driven state machine for player/NPC health and affect.

Tracks dimensions:
  - emotion:  calm -> wary -> hostile  (ordered, 3 levels)
  - trust:    0.0 ~ 1.0               (continuous float)
  - stamina:  fresh -> tired -> exhausted  (ordered, 3 levels)
  - madness:  0 ~ 100                 (integer, 100 = 失控)
  - hp:       0 ~ max_hp              (integer, 0 = dead)
  - max_hp:   derived from attributes  (integer)

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

# Madness threshold: value at which the character loses control
MAX_MADNESS: int = 100

# Madness level thresholds: <=30 sane, <=60 unstable, <=90 dangerous, >90 失控
_MADNESS_THRESHOLDS: list[tuple[int, str]] = [
    (30, "sane"),
    (60, "unstable"),
    (90, "dangerous"),
    (100, "失控"),
]

# Trigger -> list of (dimension, delta) effects
# delta for ordered dimensions is an integer index shift;
# delta for trust is a float addend.
_TRIGGER_RULES: dict[str, list[tuple[str, Any]]] = {
    "betrayed":      [("emotion", 1), ("trust", -0.1)],
    "helped":        [("emotion", -1), ("trust", 0.1)],
    "combat":        [("stamina", 1)],
    "rested":        [("stamina", -1)],
    "threatened":    [("emotion", 1)],
    "gifted":        [("trust", 0.1)],
    "use_beyonder":  [("madness", 5)],
    "horror":        [("madness", 10)],
    "acting_success": [("madness", -3)],
    "anchor_resist": [("madness", -5)],
}


def calc_max_hp(attributes: dict[str, int]) -> int:
    """Derive max HP from character attributes.

    Uses constitution*3 for LotM 8-attribute system; falls back to
    strength*2+willpower if the old 4-attribute keys are detected.
    """
    if "willpower" in attributes:
        strength = attributes.get("strength", 10)
        willpower = attributes.get("willpower", 10)
        return max(1, strength * 2 + willpower)
    constitution = attributes.get("体质", attributes.get("constitution", 10))
    return max(1, constitution * 3)


class StateMachine:
    """A lightweight, rule-driven state machine for player/NPC state.

    Parameters
    ----------
    max_hp : int
        Maximum HP (derived from character attributes via :func:`calc_max_hp`).
    emotion : str, optional
        Initial emotion level (default ``"calm"``).
    trust : float, optional
        Initial trust value between 0.0 and 1.0 (default ``0.5``).
    stamina : str, optional
        Initial stamina level (default ``"fresh"``).
    madness : int, optional
        Initial madness value (default ``0``).  Clamped to 0..MAX_MADNESS.
    """

    def __init__(
        self,
        max_hp: int,
        emotion: str = "calm",
        trust: float = 0.5,
        stamina: str = "fresh",
        madness: int = 0,
    ) -> None:
        # Validate
        if emotion not in _EMOTION_LEVELS:
            raise ValueError(f"Invalid emotion: {emotion!r}. Must be one of {_EMOTION_LEVELS}")
        if not 0.0 <= trust <= 1.0:
            raise ValueError(f"Trust must be between 0.0 and 1.0, got {trust}")
        if stamina not in _STAMINA_LEVELS:
            raise ValueError(f"Invalid stamina: {stamina!r}. Must be one of {_STAMINA_LEVELS}")
        if not 0 <= madness <= MAX_MADNESS:
            raise ValueError(f"Madness must be between 0 and {MAX_MADNESS}, got {madness}")

        self._emotion_idx = _EMOTION_LEVELS.index(emotion)
        self._trust = trust
        self._stamina_idx = _STAMINA_LEVELS.index(stamina)
        self._madness = madness
        self.max_hp = max_hp
        self.hp = max_hp

    @property
    def alive(self) -> bool:
        return self.hp > 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the current state.

        Returns
        -------
        dict
            A dictionary with keys ``emotion``, ``trust``, ``stamina``,
            ``madness``, ``madness_level``, ``hp``, ``max_hp``, ``alive``.
        """
        return {
            "emotion": _EMOTION_LEVELS[self._emotion_idx],
            "trust": round(self._trust, 2),
            "stamina": _STAMINA_LEVELS[self._stamina_idx],
            "madness": self._madness,
            "madness_level": self._derive_madness_level(),
            "hp": self.hp,
            "max_hp": self.max_hp,
            "alive": self.alive,
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
            elif dimension == "madness":
                self._madness = max(0, min(MAX_MADNESS, self._madness + delta))

        return None

    def check_madness(self) -> str:
        """Return the current madness level string.

        Returns
        -------
        str
            One of ``"sane"``, ``"unstable"``, ``"dangerous"``, ``"失控"``.
        """
        return self._derive_madness_level()

    def adjust_madness(self, delta: int) -> int:
        """Directly adjust madness by *delta*, clamped to 0..MAX_MADNESS.

        Returns the new madness value.
        """
        self._madness = max(0, min(MAX_MADNESS, self._madness + delta))
        return self._madness

    def take_damage(self, amount: int) -> str:
        """Reduce HP by *amount*.

        Returns
        -------
        str
            ``"alive"`` if still standing, ``"dead"`` if HP reached 0.
        """
        self.hp = max(0, self.hp - amount)
        return "dead" if self.hp == 0 else "alive"

    def restore_hp(self, amount: int) -> None:
        """Restore HP, capped at max_hp."""
        self.hp = min(self.max_hp, self.hp + amount)

    # ------------------------------------------------------------------
    #  Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize state to a JSON-compatible dict."""
        return {
            "hp": self.hp,
            "max_hp": self.max_hp,
            "emotion": _EMOTION_LEVELS[self._emotion_idx],
            "trust": round(self._trust, 2),
            "stamina": _STAMINA_LEVELS[self._stamina_idx],
            "madness": self._madness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateMachine":
        """Deserialize state from a dict."""
        sm = cls(
            max_hp=data.get("max_hp", 20),
            emotion=data.get("emotion", "calm"),
            trust=data.get("trust", 0.5),
            stamina=data.get("stamina", "fresh"),
            madness=data.get("madness", 0),
        )
        sm.hp = data.get("hp", sm.max_hp)
        return sm

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _derive_madness_level(self) -> str:
        for threshold, label in _MADNESS_THRESHOLDS:
            if self._madness <= threshold:
                return label
        return "失控"
