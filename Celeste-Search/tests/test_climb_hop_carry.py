"""ClimbHop's temporary moving-solid attachment and live-anchor boundaries."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, RIGHT, UP, SimState, Simulator
from celeste_search.planner import state_key
from celeste_search.runtime_state import RuntimeStateError, state_from_observation, state_readiness


def fixture():
    mover = Entity("zipMover", 1, 64, 128, 32, 16, {}, ((64.0, 80.0),))
    room = Room("a", 0, 0, 192, 184,
                ("0" * 24,) * 21 + ("1" * 24,) * 2,
                (mover,), ((24.0, 168.0),))
    return MapData("hop", {"a": room}, Path("hop.bin"))


class ClimbHopCarryTests(unittest.TestCase):
    def test_search_keys_keep_distinct_attachment_identity_and_last_position(self):
        state = SimState("a", 80, 100, force_frames=4, climb_hop_solid="a:1",
                         climb_hop_x=64, climb_hop_y=128)
        for changed in (replace(state, climb_hop_solid=""),
                        replace(state, climb_hop_solid="a:2"),
                        replace(state, climb_hop_x=65),
                        replace(state, climb_hop_y=129)):
            self.assertNotEqual(state_key(state), state_key(changed))

    def test_real_hop_remembers_platform_and_carries_airborne_without_lift_boost(self):
        sim = Simulator(fixture())
        start = SimState("a", 60, 134, vy=-15, state=1,
                         mechanisms=(("a:1", 1, 1, 64.0, 128.0, 0.0),))
        hopped = sim.step(start, GRAB | UP)
        self.assertEqual((hopped.climb_hop_solid, hopped.climb_hop_x, hopped.climb_hop_y),
                         ("a:1", 64, 128))
        self.assertEqual((hopped.force_frames, hopped.force_x), (12, 0))
        carried = sim.roll(hopped, 0, 4)
        detached = sim.roll(replace(hopped, climb_hop_solid=""), 0, 4)
        self.assertEqual(carried.y, detached.y - 1)
        self.assertEqual((carried.vy, carried.rem_y), (detached.vy, detached.rem_y))
        self.assertEqual((carried.lift_timer, carried.lift_vy), (0, 0))
        self.assertFalse(carried.on_ground)
        self.assertEqual(carried.mechanisms, detached.mechanisms)
        self.assertEqual(SimState.from_dict(carried.to_dict()), carried)
        self.assertEqual(start.climb_hop_solid, "")

    def test_expired_or_removed_attachment_cannot_keep_carrying(self):
        sim = Simulator(fixture())
        attached = SimState("a", 80, 100, force_frames=4, force_x=0,
                            climb_hop_solid="a:1", climb_hop_x=64, climb_hop_y=132,
                            mechanisms=(("a:1", 2, 0, 64.0, 128.0, 0.0),))
        for start in (replace(attached, force_frames=0),
                      replace(attached, broken=frozenset({"a:1"}))):
            with self.subTest(force=start.force_frames, broken=start.broken):
                actual = sim.step(start, 0)
                plain = sim.step(replace(start, climb_hop_solid=""), 0)
                self.assertEqual(actual.y, plain.y)
                self.assertEqual(actual.climb_hop_solid, "")

    def test_airborne_crouch_survives_coyote_time_then_releases(self):
        sim = Simulator(fixture())
        start = SimState("a", 24, 100, vx=90, vy=7.5, ducking=True,
                         grace=3, auto_jump_active=True, prev_input=RIGHT)
        first = sim.step(start, RIGHT)
        second = sim.step(first, RIGHT)
        third = sim.step(second, RIGHT)
        self.assertTrue(first.ducking)
        self.assertTrue(second.ducking)
        self.assertFalse(third.ducking)
        self.assertEqual(third.grace, 0)

    def test_runtime_cannot_invent_active_attachment_or_reuse_stale_prediction(self):
        data = fixture()
        previous = SimState("a", 80, 100, force_frames=4, force_x=0,
                            climb_hop_solid="a:1", climb_hop_x=64, climb_hop_y=132)
        model = previous.to_dict()
        for name in ("climb_hop_solid", "climb_hop_x", "climb_hop_y"):
            model.pop(name)
        observation = dict(model, model_state=model, input_phase="before_virtual_inputs")
        self.assertEqual(state_readiness(observation)["reason"], "climb_hop_attachment_unavailable")
        with self.assertRaises(RuntimeStateError):
            state_from_observation(data, observation, previous)
        settled = dict(model, force_frames=0, on_ground=True, vx=0, vy=0)
        observation = dict(settled, model_state=settled, input_phase="before_virtual_inputs")
        self.assertTrue(state_readiness(observation)["ready"])
        restored = state_from_observation(data, observation, previous)
        self.assertEqual((restored.climb_hop_solid, restored.climb_hop_x, restored.climb_hop_y),
                         ("", 0, 0))


if __name__ == "__main__":
    unittest.main()
