"""Hierarchical search regressions using only offline physics and owned maps."""

from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.maps import Entity, MapData, Room, load_map
from celeste_search.physics import Simulator
from celeste_search.planner import Goal
from celeste_search.routes import trace_route
from celeste_search.strategies import SpringGoal, search_goal


class StrategyTests(unittest.TestCase):
    def test_no_spring_falls_back_to_normal_search(self):
        rows = tuple("0" * 16 for _ in range(8)) + ("1" * 16,) * 2
        room = Room("a", 0, 0, 128, 80, rows,
                    (Entity("strawberry", 5, 64, 35, 0, 0),), ((16.0, 64.0),))
        data = MapData("fixture", {room.name: room}, Path("fixture.bin"))
        simulator = Simulator(data)
        result = search_goal(simulator, simulator.initial_state(), Goal("berry", "a", 64, 35, "a:5"),
                             deadline=time.monotonic()+3, max_expansions=1000)
        self.assertTrue(result.reached, result.reason)
        self.assertIn("a:5", result.state.berries)
        replay, _, _, _ = trace_route(simulator, "a", result.actions)
        self.assertEqual(replay, result.state)

    def test_spring_goal_requires_launch_event(self):
        rows = tuple("0" * 16 for _ in range(8)) + ("1" * 16,) * 2
        room = Room("a", 0, 0, 128, 80, rows, (), ((16.0, 64.0),))
        simulator = Simulator(MapData("fixture", {"a": room}, Path("fixture.bin")))
        state = simulator.initial_state()
        goal = SpringGoal("spring", "a", 16, 64)
        self.assertFalse(goal.achieved(state))
        self.assertTrue(goal.achieved(replace(state, auto_jump_active=True, auto_jump=0, vy=-185,y=58)))
        self.assertFalse(goal.achieved(replace(state, auto_jump_active=True, vy=-185, x=80,y=58)))

    def test_native_first_berry_continuously_from_chapter_start(self):
        path = Path(__file__).resolve().parents[2] / "GAME" / "Celeste-1" / "Content" / "Maps" / "1-ForsakenCity.bin"
        if not path.exists():
            self.skipTest("owned native map not installed")
        data = load_map(path)
        simulator = Simulator(data)
        deadline = time.monotonic()+15
        berry = search_goal(simulator, simulator.initial_state("1"), Goal("berry", "2", 336, -144, "2:11"),
                            deadline=deadline, max_expansions=5000)
        self.assertTrue(berry.reached, berry.reason)
        self.assertIn("2:11", berry.state.berries)
        self.assertLessEqual(len(berry.actions), 900)
        self.assertLessEqual(berry.expansions, 5000)
        replay, _, visited, _ = trace_route(simulator, "1", berry.actions)
        self.assertEqual(replay, berry.state)
        self.assertIn("2:11", replay.berries)
        self.assertEqual(visited, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
