"""Parity checks for branch copying; no game process is loaded or called."""
from dataclasses import FrozenInstanceError, dataclass, fields, replace
import json
import pickle
import unittest

from celeste_search.physics import SimState
from celeste_search.state_copy import evolve


class StateCopyTests(unittest.TestCase):
    def setUp(self):
        self.state = SimState("room", 17.25, -42.5, vx=90.0, vy=-105.0,
                              berries=frozenset({"room:1"}), following=frozenset({"room:2"}),
                              follow_order=("room:2",), broken=frozenset({"room:3"}),
                              fled=frozenset({"room:4"}), refill_timers=(("room:5", 61),),
                              mechanisms=(("room:6", 2, 18, 24.0, -80.0, 0.0),))

    def test_all_fields_and_combined_changes_match_dataclass_replace(self):
        for field in fields(self.state):
            with self.subTest(field=field.name):
                changes = {field.name: getattr(self.state, field.name)}
                self.assertEqual(evolve(self.state, **changes), replace(self.state, **changes))
        changes = dict(x=123.5, y=-321.25, frame=80, dash_attack=9, on_ground=True,
                       berries=self.state.berries | {"room:7"}, following=frozenset(),
                       follow_order=(), mechanisms=(("room:6", 3, 2, 90.5, -80.0, 0.0),))
        self.assertEqual(evolve(self.state, **changes), replace(self.state, **changes))

    def test_each_branch_has_independent_dictionary_and_remains_frozen(self):
        left, right = evolve(self.state, x=12), evolve(self.state, x=24)
        self.assertIsNot(left, self.state)
        self.assertIsNot(left.__dict__, right.__dict__)
        self.assertIsNot(left.__dict__, self.state.__dict__)
        with self.assertRaises(FrozenInstanceError):
            left.x = 99
        self.assertEqual(self.state.x, 17.25)
        self.assertEqual((left.x, right.x), (12, 24))
        # Shallow sharing of immutable collections is the existing semantics.
        self.assertIs(left.mechanisms, self.state.mechanisms)
        self.assertIs(left.berries, self.state.berries)

    def test_serialization_hash_and_pickle_match_replace(self):
        changes = dict(frame=987, rem_x=0.375, rem_y=-0.25, reason="copied")
        expected, actual = replace(self.state, **changes), evolve(self.state, **changes)
        self.assertEqual(hash(expected), hash(actual))
        self.assertEqual(json.dumps(expected.to_dict(), sort_keys=True), json.dumps(actual.to_dict(), sort_keys=True))
        self.assertEqual(SimState.from_dict(json.loads(json.dumps(actual.to_dict()))), expected)
        self.assertEqual(pickle.loads(pickle.dumps(actual)), expected)
        self.assertEqual(pickle.dumps(actual), pickle.dumps(expected))

    def test_empty_changes_still_produce_separate_equal_instance(self):
        actual = evolve(self.state)
        self.assertEqual(actual, replace(self.state))
        self.assertIsNot(actual, self.state)
        self.assertIsNot(actual.__dict__, self.state.__dict__)

    def test_unknown_fields_are_rejected_without_modifying_state(self):
        before = self.state.to_dict()
        for invalid in ("velcity", "__dict__"):
            with self.subTest(field=invalid), self.assertRaises(TypeError):
                evolve(self.state, **{invalid: 1})
        self.assertEqual(self.state.to_dict(), before)

    def test_unsupported_dataclass_shapes_do_not_silently_skip_initialization(self):
        @dataclass
        class Mutable:
            value: int
        @dataclass(frozen=True)
        class PostInit:
            value: int
            def __post_init__(self):
                object.__setattr__(self, "value", self.value + 1)
        @dataclass(frozen=True, slots=True)
        class Slotted:
            value: int
        for instance in (Mutable(1), PostInit(1), Slotted(1), object()):
            with self.subTest(instance=instance), self.assertRaises(TypeError):
                evolve(instance)


if __name__ == "__main__":
    unittest.main()
