"""Moving-platform staging obeys the same continuous ordered-route contract."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route
from celeste_search.assist_routes import RouteProgress
from celeste_search.assist_strategies import ZipBoardGoal, _ride_zip, _zip_ride_edges
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import SearchResult, state_key


def fixture():
    # The rising platform eventually leaves only eight pixels below the roof:
    # a full ride crushes the player, but earlier launch states are usable.
    entities = (
        Entity("zipMover", 1, 80, 96, 16, 16, {}, ((80., 40.),)),
        Entity("solid", 2, 80, 16, 16, 16),
        Entity("zipMover", 3, 128, 112, 16, 16, {}, ((128., 72.),)),
    )
    room = Room("a", 0, 0, 192, 144,
                ("0" * 24,) * 16 + ("1" * 24,) * 2, entities, ((88., 96.),))
    data = MapData("zip-invariants", {"a": room}, Path("zip-invariants.bin"))
    start = SimState("a", 88, 96, on_ground=True, frame=123,
                     mechanisms=(("a:1", 1, 7, 80., 96., 0.),
                                 ("a:3", 3, 25, 128., 72., 0.)))
    goal = ZipBoardGoal("zip_board", "a", 88, 96,
                        id="zip:a:1", identity="a:1", width=16)
    return data, Simulator(data), start, goal


def point(y):
    return {"room": "a", "x": 88, "y": y, "tile_size": 8}


def replay(simulator, start, actions):
    states = [start]
    for action in actions:
        states.append(simulator.step(states[-1], action))
    return states


class OrderedZipInvariantTests(unittest.TestCase):
    def test_safe_launches_survive_later_crush_and_preserve_full_state(self):
        _, sim, start, goal = fixture()
        original = start.to_dict()
        remembered = []
        def remember(source, arrival, states):
            remembered.append((source, arrival, list(states)))
        with patch("celeste_search.assist_strategies.remember_segment", side_effect=remember):
            edges = list(_zip_ride_edges(sim, start, goal, 100))
        self.assertEqual([len(actions) for actions, _, _ in edges], [8, 16, 24, 32])
        expected = replay(sim, start, [0] * 37)
        self.assertTrue(expected[-1].dead)
        self.assertEqual(expected[-1].reason, "squish")
        self.assertIsNone(_ride_zip(sim, start, goal, 100))
        self.assertEqual(len(remembered), len(edges))
        for (actions, arrival, label), captured in zip(edges, remembered):
            with self.subTest(label=label):
                self.assertFalse(arrival.dead)
                self.assertEqual(arrival, expected[len(actions)])
                self.assertEqual(captured, (start, arrival, expected[:len(actions) + 1]))
                self.assertEqual(arrival.frame, start.frame + len(actions))
                self.assertNotEqual(arrival.mechanisms[1], start.mechanisms[1])
        self.assertEqual(start.to_dict(), original)
        # Identical player coordinates cannot merge different moving-platform phases.
        self.assertEqual((start.x, start.y), (expected[8].x, expected[8].y))
        self.assertNotEqual(state_key(start), state_key(expected[8]))

    def test_ride_edges_obey_frame_and_deadline_limits(self):
        _, sim, start, goal = fixture()
        stats = {"simulated_frames": 10}
        edges = list(_zip_ride_edges(sim, start, goal, 20, stats=stats))
        self.assertEqual([len(actions) for actions, _, _ in edges], [8, 16])
        self.assertEqual(stats["simulated_frames"], 30)
        with patch.object(sim, "step", wraps=sim.step) as step:
            self.assertEqual(list(_zip_ride_edges(sim, start, goal, 100,
                                                 deadline=time.monotonic() - 1)), [])
            step.assert_not_called()

    def test_ordered_zip_macros_restart_at_exact_actual_waypoint_state(self):
        data, sim, start, _ = fixture()
        anchors = []

        def physical_candidate(simulator, anchor, goal, **options):
            anchors.append(anchor)
            states = [anchor]
            while len(states) < 40 and not goal.achieved(states[-1]):
                states.append(simulator.step(states[-1], 0))
                if states[-1].dead:
                    break
            actions = [0] * (len(states) - 1)
            return SearchResult(actions, states[-1], goal.achieved(states[-1]),
                                1, len(actions), "physical_test_edge")

        with patch("celeste_search.assist_planning._context", return_value=(data, sim)), \
                patch("celeste_search.assist_planning.macros", return_value=[]), \
                patch("celeste_search.assist_planning.search_zip_chain",
                      side_effect=physical_candidate), \
                patch.object(sim, "initial_state", side_effect=AssertionError("state reset")):
            result = plan_to_route("zip-invariants.bin", start.to_dict(),
                                   {"waypoints": [point(80), point(56)]},
                                   strategy="ordered_semantic", seconds=1,
                                   expansions=100, max_attempts=1)
        self.assertTrue(result["can_execute"], result)
        self.assertEqual(result["waypoint_indices"], [13, 27])
        self.assertEqual(len(anchors), 2)
        # The search normalizes only the absolute frame. All physical fields,
        # both mechanism phases, and resource timers continue unchanged.
        continuation = replay(sim, anchors[0], [0] * 13)[-1]
        self.assertEqual(anchors[1], continuation)
        actual = replay(sim, start, result["actions"])
        self.assertEqual(SimState.from_dict(result["final_state"]), actual[-1])
        self.assertFalse(any(state.dead for state in actual))

    def test_any_segment_constraint_disables_unrestricted_zip_staging(self):
        for flag in ("no_dash", "no_ground"):
            with self.subTest(flag=flag):
                data, sim, start, _ = fixture()
                with patch("celeste_search.assist_planning._context", return_value=(data, sim)), \
                        patch("celeste_search.assist_planning.macros", return_value=[]), \
                        patch("celeste_search.assist_planning.search_zip_chain") as recipe:
                    result = plan_to_route("zip-invariants.bin", start.to_dict(),
                              {"waypoints": [point(80), point(56)], "segments": [{}, {flag: True}]},
                              strategy="ordered_semantic", seconds=1,
                              expansions=5, max_attempts=1)
                recipe.assert_not_called()
                self.assertFalse(result["can_execute"])
                self.assertEqual(result["actions"], [])

    def test_claimed_complete_prefix_or_wrong_order_never_becomes_executable(self):
        cases = (([point(80), point(56)], 13),
                 ([point(56), point(80)], 27))
        for points, length in cases:
            with self.subTest(points=points, length=length):
                data, sim, start, _ = fixture()
                actions = [0] * length
                end = replay(sim, start, actions)[-1]
                candidate = (actions, end, True, 1, length, RouteProgress(2), {})
                with patch("celeste_search.assist_planning._context", return_value=(data, sim)), \
                        patch("celeste_search.assist_planning._search_ordered_route", return_value=candidate):
                    result = plan_to_route("zip-invariants.bin", start.to_dict(),
                                           {"waypoints": points}, strategy="ordered_semantic",
                                           seconds=1, max_attempts=1)
                self.assertFalse(result["can_execute"])
                self.assertEqual(result["actions"], [])
                self.assertTrue(result["attempts"][0]["replay_rejected"])

    def test_grounded_ride_cannot_bypass_protected_second_segment(self):
        data, sim, start, _ = fixture()
        actions = [0] * 27
        end = replay(sim, start, actions)[-1]
        candidate = (actions, end, True, 1, len(actions), RouteProgress(2), {})
        with patch("celeste_search.assist_planning._context", return_value=(data, sim)), \
                patch("celeste_search.assist_planning._search_ordered_route", return_value=candidate):
            result = plan_to_route("zip-invariants.bin", start.to_dict(),
                      {"waypoints": [point(80), point(56)], "segments": [{}, {"no_ground": True}]},
                      strategy="ordered_semantic", seconds=1, max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"], [])
        self.assertTrue(result["attempts"][0]["replay_rejected"])


if __name__ == "__main__":
    unittest.main()
