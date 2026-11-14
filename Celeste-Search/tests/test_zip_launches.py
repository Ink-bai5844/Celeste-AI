"""Lift launches use real support, steering and resource transitions."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_strategies import ZipBoardGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import Simulator, SimState
from celeste_search.zip_launches import zip_launch_edges


def fixture(mirror=False, ox=0, oy=0):
    width = 224
    rows = [["0"] * (width // 8) for _ in range(24)]
    for left, right, bottom in ((48, 80, 96), (80, 88, 48),
                                (88, 120, 8), (120, 128, 48), (128, 160, 96)):
        for y in range(bottom // 8):
            for x in range(left // 8, right // 8):
                rows[y][x] = "1"
    source_entities = (
        Entity("zipMover", 1, 48, 144, 32, 16, {}, ((48., 96.),)),
        Entity("spikesRight", 2, 88, 8, 0, 48),
        Entity("spikesLeft", 3, 120, 8, 0, 40),
        Entity("spikesDown", 4, 96, 8, 16, 0),
        Entity("zipMover", 5, 192, 160, 16, 16, {}, ((192., 128.),)),
    )
    entities = []
    for entity in source_entities:
        x = width - entity.x - entity.width if mirror else entity.x
        kind = ({"spikesLeft": "spikesRight", "spikesRight": "spikesLeft"}
                .get(entity.kind, entity.kind) if mirror else entity.kind)
        nodes = tuple((ox + (width - x - entity.width if mirror else x), oy + y)
                      for x, y in entity.nodes)
        entities.append(replace(entity, kind=kind, x=x + ox, y=entity.y + oy, nodes=nodes))
    solids = tuple("".join(reversed(row) if mirror else row) for row in rows)
    room = Room("a", ox, oy, width, 192, solids, tuple(entities), ((ox + 16., oy + 180.),))
    data = MapData("zip-launch", {"a": room}, Path("zip-launch.bin"))
    mover, other = entities[0], entities[-1]
    direction = -1 if mirror else 1
    start = SimState("a", ox + (width - 52 if mirror else 52), oy + 133,
                     vx=33.333396911621094 * direction, dashes=1, on_ground=True,
                     frame=123, facing=direction, prev_input=2 if mirror else 1,
                     rem_x=-.2916678190231323 * direction, grace=6,
                     lift_vy=-91.78693389892578, lift_timer=10,
                     mechanisms=(("a:1", 2, 13, mover.x, oy + 133.302978515625, 0.),
                                 ("a:5", 3, 7, other.x, oy + 128., 0.)))
    board = ZipBoardGoal("zip_board", "a", mover.x + 16, mover.y,
                         identity="a:1", width=32)
    goal = _selected_goal({"room": "a", "x": ox + (120 if mirror else 96),
                           "y": oy + 24})[0]
    return Simulator(data), start, board, goal


class ZipLaunchTests(unittest.TestCase):
    def test_mirrored_translated_overhang_routes_replay_with_lift_and_dash_refill(self):
        for mirror, ox, oy in ((False, 0, 0), (False, -320, -480),
                               (True, 0, 0), (True, -320, -480)):
            with self.subTest(mirror=mirror, offset=(ox, oy)):
                sim, start, board, goal = fixture(mirror, ox, oy)
                original = start.to_dict()
                edges = list(zip_launch_edges(sim, start, board, goal,
                                               deadline=time.monotonic() + 2))
                self.assertTrue(edges)
                for actions, arrival, _ in edges:
                    state = start
                    for action in actions:
                        state = sim.step(state, action)
                        self.assertFalse(state.dead)
                    self.assertEqual(state, arrival)
                    self.assertTrue(goal.achieved(arrival))
                    self.assertEqual(arrival.frame, start.frame + len(actions))
                    self.assertNotEqual(arrival.mechanisms[1], start.mechanisms[1])
                actions = edges[0][0]
                states = [start]
                for action in actions:
                    states.append(sim.step(states[-1], action))
                self.assertTrue(any(before.dashes == 0 and after.dashes == 1
                                    for before, after in zip(states, states[1:])))
                self.assertTrue(any(state.vy < -150 and state.lift_timer > 0 for state in states))
                self.assertEqual(start.to_dict(), original)

    def test_full_preview_prefixes_and_local_counters_are_preserved(self):
        sim, start, board, goal = fixture()
        captured = []
        tracker = Mock()
        stats = {"expansions": 10, "simulated_frames": 500, "max_expansions": 12}
        with patch("celeste_search.zip_launches.search_tracker", return_value=tracker), \
                patch("celeste_search.zip_launches.remember_segment",
                      side_effect=lambda first, last, states: captured.append((first, last, list(states)))):
            edges = list(zip_launch_edges(sim, start, board, goal,
                                           deadline=time.monotonic() + 2, stats=stats))
        self.assertTrue(edges)
        self.assertEqual(stats["expansions"], 12)
        self.assertEqual(len(captured), len(edges))
        for (actions, arrival, _), (first, last, states) in zip(edges, captured):
            self.assertEqual((first, last), (start, arrival))
            self.assertEqual(len(states), len(actions) + 1)
            for before, action, after in zip(states, actions, states[1:]):
                self.assertEqual(sim.step(before, action), after)
        self.assertEqual(max(call.args[1] for call in tracker.report.call_args_list), 2)
        self.assertEqual(max(call.args[2] for call in tracker.report.call_args_list),
                         stats["simulated_frames"] - 500)

    def test_deadline_frame_and_expansion_limits_do_not_yield_partial_hits(self):
        sim, start, board, goal = fixture()
        with patch.object(sim, "step", wraps=sim.step) as step:
            self.assertEqual(list(zip_launch_edges(sim, start, board, goal,
                                deadline=time.monotonic() - 1)), [])
            for frames in (0, -1):
                self.assertEqual(list(zip_launch_edges(sim, start, board, goal,
                                     deadline=time.monotonic() + 1, max_frames=frames)), [])
            self.assertEqual(list(zip_launch_edges(sim, start, board, goal,
                                 deadline=time.monotonic() + 1, stats={"max_expansions": -1})), [])
            step.assert_not_called()
        stats = {"max_expansions": 1}
        self.assertEqual(list(zip_launch_edges(sim, start, board, goal,
                             deadline=time.monotonic() + 2, max_frames=5, stats=stats)), [])
        self.assertEqual(stats["expansions"], 1)
        self.assertLessEqual(stats["simulated_frames"], 5)

    def test_dead_and_unrelated_room_successors_are_rejected(self):
        for changed in ({"dead": True}, {"room": "unrelated"}):
            with self.subTest(changed=changed):
                sim, start, board, goal = fixture()
                with patch.object(sim, "step", return_value=replace(start, **changed)):
                    self.assertEqual(list(zip_launch_edges(sim, start, board, goal,
                                     deadline=time.monotonic() + 1)), [])
        sim, start, board, goal = fixture()
        with patch.object(sim, "step", wraps=sim.step) as step:
            self.assertEqual(list(zip_launch_edges(sim, replace(start, on_ground=False), board, goal,
                                 deadline=time.monotonic() + 1)), [])
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
