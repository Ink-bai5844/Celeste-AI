"""Cassette support targets and waiting edges retain live beat state."""
from dataclasses import replace
import time
import unittest

from celeste_search.assist_strategies import _support_candidates, cassette_wait_edges
from celeste_search.geometry_milestones import cassette_support_goals, transient_support_goals
from celeste_search.physics import Simulator
from celeste_search.planner import Goal
from celeste_search.runtime_state import state_from_observation
from test_cassette_runtime import fixture, observed


class CassetteMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.state = state_from_observation(self.data, observed())
        self.simulator = Simulator(self.data)
        self.room = self.data.rooms["a"]

    def test_inventory_requires_observed_phase_and_keeps_inactive_future_supports(self):
        self.assertEqual(cassette_support_goals(self.room, replace(self.state, cassette_clock=None)), [])
        blocks = tuple(replace(block, activated=False, collidable=False) for block in self.state.cassette_blocks)
        inactive = replace(self.state, cassette_blocks=blocks)
        goals = cassette_support_goals(self.room, inactive)
        self.assertEqual(len(goals), 2)
        self.assertEqual(len(transient_support_goals(self.room, inactive)), 2)
        self.assertTrue(all(goal.requires_continuation and not goal.idle_endpoint_allowed for goal in goals))

    def test_landing_uses_current_collidability_and_shifted_geometry(self):
        goal = cassette_support_goals(self.room, self.state)[0]
        landed = replace(self.state, x=56.0, y=48.0, on_ground=True, vy=0.0)
        self.assertTrue(goal.achieved(landed))
        shifted = replace(landed, cassette_blocks=(replace(landed.cassette_blocks[0], y=49.0), landed.cassette_blocks[1]))
        self.assertFalse(goal.achieved(shifted))
        self.assertEqual(goal.waypoint(shifted), (56.0, 49.0))
        disabled = replace(landed, cassette_blocks=(replace(landed.cassette_blocks[0], collidable=False), landed.cassette_blocks[1]))
        self.assertFalse(goal.achieved(disabled))
        self.assertFalse(goal.achieved(replace(landed, on_ground=False)))
        block=landed.cassette_blocks[0]
        self.assertTrue(goal.achieved(replace(landed,x=block.x-2)))
        self.assertFalse(goal.achieved(replace(landed,x=block.x-4)))

    def test_support_graph_includes_dynamic_block_candidates(self):
        target = Goal("point", "a", 100.0, 20.0)
        candidates = _support_candidates(self.simulator, self.state, target, "forward", ())
        self.assertTrue(any(key == "cassette_support:a:1" for _, key, _ in candidates))

    def test_waits_are_actual_neutral_replays_with_continuous_phase(self):
        stats={}
        edges = list(cassette_wait_edges(self.simulator, self.state,
                                        deadline=time.monotonic()+2, max_frames=160,stats=stats))
        self.assertTrue(edges)
        self.assertLessEqual(stats['simulated_frames'],160)
        for actions, arrival, _ in edges:
            self.assertTrue(actions)
            self.assertEqual(set(actions), {0})
            replay = self.state
            for mask in actions:
                replay = self.simulator.step(replay, mask)
            self.assertEqual(replay.to_dict(), arrival.to_dict())
            self.assertEqual(arrival.frame, self.state.frame+len(actions))
            self.assertFalse(arrival.dead)
            self.assertNotEqual(arrival.cassette_clock, self.state.cassette_clock)


if __name__ == "__main__":
    unittest.main()
