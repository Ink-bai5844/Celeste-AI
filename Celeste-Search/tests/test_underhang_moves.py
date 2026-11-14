"""Geometry-driven underhang routes retain exact simulated states and bounds."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.underhang_moves import underhang_edges, underhang_roofs


def fixture(mirrored=False):
    entities = [Entity("crumbleBlock", 1, 64, 64, 24, 0),
                Entity("jumpThru", 2, 56, 96, 24, 0),
                Entity("spikesUp", 3, 56, 96, 32, 0),
                Entity("solid", 4, 80, 96, 16, 48)]
    if mirrored:
        entities = [replace(entity, x=160-entity.x-entity.width) for entity in entities]
    room = Room("a", 0, 0, 160, 160, ("0"*20,)*18+("1"*20,)*2, tuple(entities), ())
    sim = Simulator(MapData("underhang-fixture", {"a": room}, Path("fixture.bin")))
    state = SimState("a", 88 if mirrored else 72, 64, vx=90 if mirrored else -90,
                     on_ground=True, mechanisms=(("a:1", 1, 0, 72. if mirrored else 64., 64., 0.),))
    goal, _ = _selected_goal({"room": "a", "x": 80 if mirrored else 72, "y": 104})
    return sim, state, goal


class UnderhangMoveTests(unittest.TestCase):
    def test_both_sides_physically_drop_reverse_and_reach_exact_tile(self):
        for mirrored in (False, True):
            with self.subTest(mirrored=mirrored):
                sim, start, goal = fixture(mirrored)
                before = start.to_dict()
                stats = {"expansions": 7, "simulated_frames": 13}
                with patch("celeste_search.underhang_moves.remember_segment") as remember:
                    edges = list(underhang_edges(sim, start, goal, deadline=time.monotonic()+2, stats=stats))
                self.assertTrue(edges)
                self.assertEqual(remember.call_count, len(edges))
                attempted_frames = stats["simulated_frames"]-13
                self.assertGreaterEqual(attempted_frames, sum(len(actions) for actions, _, _ in edges))
                self.assertEqual(stats["expansions"], 9)
                for (actions, end, label), call in zip(edges, remember.call_args_list):
                    trace = sim.trace(start, actions)
                    self.assertEqual(trace[-1], end)
                    self.assertTrue(goal.achieved(end))
                    self.assertTrue(all(not state.dead and state.room == start.room for state in trace))
                    self.assertEqual(call.args, (start, end, trace))
                    self.assertEqual(end.frame-start.frame, len(actions))
                    self.assertNotEqual(end.mechanisms, start.mechanisms)
                    self.assertIn("right" if mirrored else "left", label)
                    self.assertTrue(any(action & DASH for action in actions))
                self.assertEqual(start.to_dict(), before)

    def test_dash_waits_for_whole_body_below_roof_not_just_target_row(self):
        sim, start, goal = fixture()
        roof = underhang_roofs(sim, start, goal)[0]
        self.assertEqual((roof["left"], roof["right"], roof["bottom"]), (56, 88, 101))
        actions, end, _ = next(underhang_edges(sim, start, goal, deadline=time.monotonic()+2))
        trace = sim.trace(start, actions)
        dash_index = next(index for index, action in enumerate(actions) if action & DASH)
        before_dash = trace[dash_index]
        self.assertGreaterEqual(before_dash.y-11, roof["bottom"])
        self.assertGreaterEqual(before_dash.y-5.5, goal.tile_y)
        self.assertLess(before_dash.y-5.5, goal.tile_y+goal.tile_size)
        self.assertEqual(end.y, before_dash.y)

    def test_deadline_frame_budget_and_room_change_cannot_yield_partial_success(self):
        sim, start, goal = fixture()
        with patch.object(sim, "step", side_effect=AssertionError("expired work must not step")):
            self.assertEqual(list(underhang_edges(sim, start, goal, deadline=time.monotonic()-1)), [])
        stats = {}
        self.assertEqual(list(underhang_edges(sim, start, goal, deadline=time.monotonic()+2,
                                             max_frames=5, stats=stats)), [])
        self.assertLessEqual(stats["simulated_frames"], 10)
        calls = [0]
        def ticking_clock():
            calls[0] += 1
            return 0 if calls[0] < 8 else 2
        stats = {}
        with patch("celeste_search.underhang_moves.time.monotonic", side_effect=ticking_clock):
            self.assertEqual(list(underhang_edges(sim, start, goal, deadline=1, stats=stats)), [])
        self.assertLessEqual(stats["simulated_frames"], 6)
        original = sim.step
        for updates in ({"room": "b"}, {"dead": True}):
            with patch.object(sim, "step", side_effect=lambda state, mask: replace(original(state, mask), **updates)), \
                    patch("celeste_search.underhang_moves.remember_segment") as remember:
                self.assertEqual(list(underhang_edges(sim, start, goal, deadline=time.monotonic()+2)), [])
                remember.assert_not_called()

    def test_shared_expansion_budget_is_checked_before_next_branch(self):
        sim, start, goal = fixture()
        stats = {"expansions": 7, "simulated_frames": 13, "max_expansions": 8}
        edges = list(underhang_edges(sim, start, goal, deadline=time.monotonic()+2, stats=stats))
        self.assertTrue(edges)
        self.assertEqual(stats["expansions"], 8)
        with patch.object(sim, "step", side_effect=AssertionError("budget exhausted")):
            self.assertEqual(list(underhang_edges(sim, start, goal, deadline=time.monotonic()+2, stats=stats)), [])

    def test_roofs_reject_other_rooms_exits_and_tiles_not_below_platform(self):
        sim, start, goal = fixture()
        self.assertEqual(underhang_roofs(sim, replace(start, dead=True), goal), [])
        self.assertEqual(underhang_roofs(sim, start, replace(goal, room="b", source_room="b")), [])
        self.assertEqual(underhang_roofs(sim, start, replace(goal, transition_to="b")), [])
        self.assertEqual(underhang_roofs(sim, start, replace(goal, tile_y=64)), [])
        self.assertEqual(underhang_roofs(sim, start, replace(goal, tile_x=120)), [])


if __name__ == "__main__":
    unittest.main()
