"""Future input boundaries and planned room crossings; all game state is fake."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from celeste_search.background import BackgroundRoomPlanner
from celeste_search.exploration import Exploration
from celeste_search.live import LiveController
from celeste_search.maps import MapData, Room
from celeste_search.periodic import projected_anchor_matches
from celeste_search.physics import JUMP, RIGHT, SimState, Simulator
from test_background import FakeExecutor, candidate as background_candidate
from test_live import FakeBackground, raw_observation


def observation(state, frame, **kwargs):
    result = raw_observation(state, frame, **kwargs)
    result["model_state"] = state.to_dict()
    return result


class LiveNavigationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "navigation.bin"
        source.write_bytes(b"navigation fixture")
        rows = ("0" * 40,) * 8 + ("1" * 40,) * 2
        room = Room("a", 0, 0, 320, 80, rows, (), ((315.0, 64.0),))
        other = replace(room, name="b", x=320, spawns=((344.0, 64.0),))
        self.map = MapData("fixture", {"a": room, "b": other}, source)
        self.sim = Simulator(self.map)
        self.start = self.sim.initial_state("a")

    def primed(self):
        background = FakeBackground(self.map)
        controller = LiveController(self.map, background)
        for frame in range(4):
            self.assertEqual(controller.observe(observation(self.start, frame)), 0)
        self.assertIsNotNone(controller.pending_anchor)
        return controller, background

    def test_planned_airborne_crossing_keeps_next_jump_input_without_grounding(self):
        controller, background = self.primed()
        candidate, states = background.candidate([RIGHT | JUMP] * 20)
        candidate["allow_room_continuation"] = True
        background.publish(candidate)
        self.assertEqual(controller.observe(observation(self.start, 4)), RIGHT | JUMP)
        crossing = next(index for index, state in enumerate(states) if state.room == "b")
        self.assertFalse(states[crossing].on_ground)
        for index in range(1, crossing + 1):
            self.assertEqual(controller.observe(observation(states[index], 4 + index)), RIGHT | JUMP)
        self.assertIs(controller.plan, candidate)
        self.assertEqual(controller.index, crossing + 1)
        event = [event for event in controller.events if event["event"] == "room_entered"][-1]
        self.assertTrue(event["planned_continuation"])
        self.assertEqual(controller.visited, {"a", "b"})

    def test_unplanned_crossing_releases_stale_route(self):
        controller, background = self.primed()
        candidate, states = background.candidate([RIGHT | JUMP] * 20)
        background.publish(candidate)
        controller.observe(observation(self.start, 4))
        crossing = next(index for index, state in enumerate(states) if state.room == "b")
        for index in range(1, crossing):
            controller.observe(observation(states[index], 4 + index))
        self.assertEqual(controller.observe(observation(states[crossing], 4 + crossing)), 0)
        self.assertIsNone(controller.plan)
        self.assertEqual(controller.index, 0)
        self.assertFalse([event for event in controller.events if event["event"] == "room_entered"][-1]["planned_continuation"])

    def test_spawn_change_cannot_inherit_planned_crossing_inputs(self):
        controller, background = self.primed()
        candidate, states = background.candidate([RIGHT | JUMP] * 20)
        candidate["allow_room_continuation"] = True
        background.publish(candidate)
        controller.observe(observation(self.start, 4))
        crossing = next(index for index, state in enumerate(states) if state.room == "b")
        for index in range(1, crossing):
            controller.observe(observation(states[index], 4 + index))
        self.assertEqual(controller.observe(observation(states[crossing], 4 + crossing, spawn=2, deaths=1)), 0)
        self.assertIsNone(controller.plan)
        self.assertTrue(any(event["event"] == "natural_respawn" for event in controller.events))

    def test_changed_navigation_revision_rejects_even_matching_idle_anchor(self):
        controller, background = self.primed()
        candidate, _ = background.candidate()
        candidate["navigation_revision"] = controller.exploration.navigation_revision
        controller.exploration.set_destination("b")
        background.publish(candidate)
        self.assertEqual(controller.observe(observation(self.start, 4)), 0)
        self.assertIsNone(controller.plan)
        self.assertTrue(any(event["event"] == "stale_plan_discarded" for event in controller.events))
        self.assertEqual(controller.room_retries, 0)

    def test_future_candidate_arriving_after_absolute_frame_is_discarded(self):
        controller, background = self.primed()
        controller.pending_kind = "periodic"
        candidate, _ = background.candidate()
        future = replace(self.start, frame=4)
        candidate.update(execute_at_frame=4, execution_anchor_state=future.to_dict(),
                         requires_future_anchor=True, requires_periodic_anchor=False)
        self.assertEqual(controller.observe(observation(self.start, 4)), 0)
        background.publish(candidate)
        self.assertEqual(controller.observe(observation(self.start, 5)), 0)
        self.assertIsNone(controller.plan)
        self.assertIsNone(controller.waiting_candidate)
        self.assertEqual(controller.room_retries, 0)

    def test_future_matching_rejects_subpixels_and_motion_private_fields(self):
        rising = replace(self.start, on_ground=False, y=40, vy=-185, var_jump=12,
                         var_speed=-185, auto_jump_active=True, rem_x=.125, rem_y=.456)
        actual = observation(rising, 800)
        origin = observation(rising, 600)
        expected = rising.to_dict()
        self.assertTrue(projected_anchor_matches(actual, expected, origin))
        for key in ("rem_x", "rem_y", "var_speed", "auto_jump", "grace", "force_x", "force_frames",
                    "wall_retention", "wall_retained", "hop_wait_x", "hop_wait_x_speed",
                    "lift_timer", "lift_vx", "lift_vy", "max_fall", "dash_attack"):
            changed = deepcopy(actual)
            changed["model_state"][key] += .25 if key.startswith("rem_") else 1
            with self.subTest(field=key):
                self.assertFalse(projected_anchor_matches(changed, expected, origin))
        changed = dict(actual, stamina=100)
        self.assertFalse(projected_anchor_matches(changed, expected, origin))


class NavigationBatchTests(unittest.TestCase):
    def test_background_stamps_captured_revision_and_preserves_future_anchor(self):
        with patch("celeste_search.background.ProcessPoolExecutor", FakeExecutor):
            planner = BackgroundRoomPlanner("fixture.bin", workers=1)
        self.addCleanup(planner.close)
        original = SimState("a", 24, 64, on_ground=True, frame=10)
        exploration = {"navigation_revision": 7}
        self.assertTrue(planner.request(original, exploration=exploration, recovery=True))
        exploration["navigation_revision"] = 8
        future = replace(original, frame=210, rem_y=.456).to_dict()
        result = background_candidate()
        result.update(execution_anchor_state=future, execute_at_frame=210, requires_future_anchor=True)
        planner._executor.submissions[0][0].set_result(result)
        published = planner.poll()
        self.assertEqual(published["navigation_revision"], 7)
        self.assertEqual(SimState.from_dict(published["anchor_state"]), original)
        self.assertEqual(published["execution_anchor_state"], future)
        self.assertEqual(published["execute_at_frame"], 210)

    def test_destination_revision_changes_only_on_actual_choice_or_settling(self):
        ledger = Exploration()
        ledger.set_destination("b")
        revision = ledger.navigation_revision
        ledger.set_destination("b")
        ledger.mark_settled("a")
        self.assertEqual(ledger.navigation_revision, revision)
        ledger.mark_settled("b")
        self.assertEqual(ledger.navigation_revision, revision + 1)
        self.assertIsNone(ledger.active_destination)
        restored = Exploration(ledger.snapshot())
        self.assertEqual(restored.navigation_revision, ledger.navigation_revision)


if __name__ == "__main__":
    unittest.main()
