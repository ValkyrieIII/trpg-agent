"""Tests for the state module — rule-driven state machine for emotion/trust/stamina."""

from trpg_agent.state import StateMachine


class TestInitialState:
    """Default state values after construction."""

    def test_initial_emotion_is_calm(self):
        sm = StateMachine(max_hp=20)
        state = sm.get_state()
        assert state["emotion"] == "calm"

    def test_initial_trust_is_0_5(self):
        sm = StateMachine(max_hp=20)
        state = sm.get_state()
        assert state["trust"] == 0.5

    def test_initial_stamina_is_fresh(self):
        sm = StateMachine(max_hp=20)
        state = sm.get_state()
        assert state["stamina"] == "fresh"

    def test_get_state_returns_copy(self):
        """get_state should return a new dict, not the internal one."""
        sm = StateMachine(max_hp=20)
        state1 = sm.get_state()
        state2 = sm.get_state()
        assert state1 is not state2


class TestEmotionRising:
    """Emotion ascending path: calm -> wary -> hostile."""

    def test_calm_to_wary(self):
        sm = StateMachine(max_hp=20)
        sm.apply("threatened")
        assert sm.get_state()["emotion"] == "wary"

    def test_wary_to_hostile(self):
        sm = StateMachine(max_hp=20)
        sm.apply("threatened")
        sm.apply("betrayed")  # betrayed also does +1
        assert sm.get_state()["emotion"] == "hostile"

    def test_hostile_stays_hostile(self):
        sm = StateMachine(max_hp=20)
        sm.apply("threatened")
        sm.apply("betrayed")
        sm.apply("betrayed")  # already hostile, should stay hostile
        assert sm.get_state()["emotion"] == "hostile"

    def test_betrayed_raises_emotion(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")
        assert sm.get_state()["emotion"] == "wary"

    def test_threatened_raises_emotion(self):
        sm = StateMachine(max_hp=20)
        sm.apply("threatened")
        assert sm.get_state()["emotion"] == "wary"


class TestEmotionDescending:
    """Emotion descending path: hostile -> wary -> calm."""

    def test_hostile_to_wary(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")
        sm.apply("betrayed")  # now hostile
        sm.apply("helped")     # helped does emotion-1
        assert sm.get_state()["emotion"] == "wary"

    def test_wary_to_calm(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")  # now wary
        sm.apply("helped")    # back to calm
        assert sm.get_state()["emotion"] == "calm"

    def test_calm_stays_calm(self):
        sm = StateMachine(max_hp=20)
        sm.apply("helped")  # calm -1 should stay calm
        assert sm.get_state()["emotion"] == "calm"


class TestTrustChanges:
    """Trust increase, decrease, and boundary clamping."""

    def test_helped_increases_trust(self):
        sm = StateMachine(max_hp=20)
        sm.apply("helped")
        assert sm.get_state()["trust"] == 0.6

    def test_gifted_increases_trust(self):
        sm = StateMachine(max_hp=20)
        sm.apply("gifted")
        assert sm.get_state()["trust"] == 0.6

    def test_betrayed_decreases_trust(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")
        assert sm.get_state()["trust"] == 0.4

    def test_trust_clamps_at_1_0(self):
        sm = StateMachine(max_hp=20)
        for _ in range(20):
            sm.apply("helped")
        assert sm.get_state()["trust"] == 1.0

    def test_trust_clamps_at_0_0(self):
        sm = StateMachine(max_hp=20)
        for _ in range(20):
            sm.apply("betrayed")
        assert sm.get_state()["trust"] == 0.0

    def test_trust_at_1_0_stays_1_0(self):
        sm = StateMachine(max_hp=20)
        for _ in range(20):
            sm.apply("helped")
        # extra helps should not lower it
        sm.apply("helped")
        assert sm.get_state()["trust"] == 1.0

    def test_trust_at_0_0_stays_0_0(self):
        sm = StateMachine(max_hp=20)
        for _ in range(20):
            sm.apply("betrayed")
        # extra betrayals should not raise it
        sm.apply("betrayed")
        assert sm.get_state()["trust"] == 0.0


class TestStaminaChanges:
    """Stamina consumption and recovery, with boundary clamping."""

    def test_combat_reduces_stamina(self):
        sm = StateMachine(max_hp=20)
        sm.apply("combat")
        assert sm.get_state()["stamina"] == "tired"

    def test_combat_twice_exhausted(self):
        sm = StateMachine(max_hp=20)
        sm.apply("combat")
        sm.apply("combat")
        assert sm.get_state()["stamina"] == "exhausted"

    def test_combat_thrice_stays_exhausted(self):
        sm = StateMachine(max_hp=20)
        sm.apply("combat")
        sm.apply("combat")
        sm.apply("combat")
        assert sm.get_state()["stamina"] == "exhausted"

    def test_rested_recovers_stamina(self):
        sm = StateMachine(max_hp=20)
        sm.apply("combat")  # tired
        sm.apply("rested")  # back to fresh
        assert sm.get_state()["stamina"] == "fresh"

    def test_rested_from_exhausted(self):
        sm = StateMachine(max_hp=20)
        sm.apply("combat")
        sm.apply("combat")  # exhausted
        sm.apply("rested")  # back to tired
        assert sm.get_state()["stamina"] == "tired"

    def test_fresh_stays_fresh_after_rest(self):
        sm = StateMachine(max_hp=20)
        sm.apply("rested")  # fresh stays fresh
        assert sm.get_state()["stamina"] == "fresh"


class TestBoundaryClamping:
    """All dimensions clamped at their bounds."""

    def test_emotion_upper_bound(self):
        sm = StateMachine(max_hp=20)
        for _ in range(10):
            sm.apply("betrayed")
        assert sm.get_state()["emotion"] == "hostile"

    def test_emotion_lower_bound(self):
        sm = StateMachine(max_hp=20)
        for _ in range(10):
            sm.apply("helped")
        assert sm.get_state()["emotion"] == "calm"

    def test_stamina_upper_bound(self):
        sm = StateMachine(max_hp=20)
        for _ in range(10):
            sm.apply("combat")
        assert sm.get_state()["stamina"] == "exhausted"

    def test_stamina_lower_bound(self):
        sm = StateMachine(max_hp=20)
        for _ in range(10):
            sm.apply("rested")
        assert sm.get_state()["stamina"] == "fresh"


class TestUnknownTrigger:
    """Unknown triggers are silently ignored."""

    def test_unknown_trigger_returns_none_and_no_change(self):
        sm = StateMachine(max_hp=20)
        result = sm.apply("unknown_trigger")
        assert result is None
        assert sm.get_state()["emotion"] == "calm"
        assert sm.get_state()["trust"] == 0.5
        assert sm.get_state()["stamina"] == "fresh"

    def test_unknown_trigger_does_not_affect_state(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")
        state_before = sm.get_state()
        sm.apply("nonexistent")
        assert sm.get_state() == state_before

    def test_empty_trigger_silently_ignored(self):
        sm = StateMachine(max_hp=20)
        sm.apply("")
        assert sm.get_state()["emotion"] == "calm"
        assert sm.get_state()["trust"] == 0.5
        assert sm.get_state()["stamina"] == "fresh"


class TestCombinedTriggers:
    """Combinations of triggers produce correct aggregate state."""

    def test_betrayed_full_effect(self):
        """betrayed: emotion+1, trust-0.1."""
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")
        assert sm.get_state()["emotion"] == "wary"
        assert sm.get_state()["trust"] == 0.4
        assert sm.get_state()["stamina"] == "fresh"

    def test_helped_full_effect(self):
        """helped: emotion-1, trust+0.1."""
        sm = StateMachine(max_hp=20)
        # first raise emotion so it can go down
        sm.apply("threatened")
        sm.apply("helped")
        assert sm.get_state()["emotion"] == "calm"
        assert sm.get_state()["trust"] == 0.6
        assert sm.get_state()["stamina"] == "fresh"

    def test_multiple_triggers_accumulate(self):
        sm = StateMachine(max_hp=20)
        sm.apply("betrayed")   # wary, 0.4
        sm.apply("helped")     # calm, 0.5
        sm.apply("combat")     # calm, 0.5, tired
        sm.apply("gifted")     # calm, 0.6, tired
        assert sm.get_state()["emotion"] == "calm"
        assert sm.get_state()["trust"] == 0.6
        assert sm.get_state()["stamina"] == "tired"
