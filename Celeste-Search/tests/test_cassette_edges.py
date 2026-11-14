"""Cassette search templates emit only uninterrupted physical landing edges."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.cassette import CassetteBlockState, CassetteClock
from celeste_search.cassette_edges import cassette_launch_edges, cassette_support_edges
from celeste_search.geometry_milestones import cassette_support_goals
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import JUMP, SimState, Simulator


def fixture(facing=1):
    block_x = 64. if facing > 0 else 32.
    player_x = 16. if facing > 0 else 112.
    entity = Entity("cassetteBlock", 1, block_x, 40., 32., 16., {"index": 0})
    room = Room("a", 0, 0, 128, 80, ("0"*16,)*8+("1"*16,)*2, (entity,), ())
    sim = Simulator(MapData("cassette-edges", {"a": room}, Path("fixture.bin")))
    clock = CassetteClock(0, 0., 0, 1., 0, 2, True, 0, True, True, 0,
                          actual_depth=-.000001,
                          depth_counters=((-10,.000001),(-5,.000001),(0,.000003),
                                          (10,.000001),(15,.000001),(8990,.000001),(8995,.000001)))
    block = CassetteBlockState("a:1", 0, 1., block_x, 40., 32., 16.,
                               True, True, True, 2, True, ("a:1",), 2, 0., 0., 0.,
                               depth=-10, actual_depth=-10., tracker_order=0)
    start = SimState("a", player_x, 64., on_ground=True, grace=6, prev_input=JUMP,
                     cassette_clock=clock, cassette_blocks=(block,),
                     player_update_order=1, player_actual_depth=-.000002)
    return sim, start, cassette_support_goals(room, start)[0]


class CassetteEdgeTests(unittest.TestCase):
    def test_relative_jump_landings_replay_both_directions_with_live_phase(self):
        for facing in (1, -1):
            sim, start, goal = fixture(facing)
            original = start.to_dict()
            stats = {}
            edges = list(cassette_support_edges(sim, start, goal, stats=stats))
            self.assertTrue(edges)
            self.assertGreater(stats["simulated_frames"], 0)
            self.assertGreater(stats["expansions"], 0)
            for actions, end, label in edges:
                self.assertEqual(actions[0], 0, label)
                self.assertEqual(sim.trace(start, actions)[-1], end, label)
                self.assertTrue(goal.achieved(end), label)
                self.assertNotEqual(start.cassette_clock, end.cassette_clock)
                self.assertEqual(end.frame-start.frame, len(actions))
            self.assertEqual(start.to_dict(), original)

    def test_edge_walk_stops_before_losing_support_and_keeps_full_replay(self):
        sim, start, goal = fixture()
        landing = next(cassette_support_edges(sim, start, goal))[1]
        edges = list(cassette_launch_edges(sim, landing, goal))
        self.assertEqual({label for _, _, label in edges}, {"left_edge", "right_edge"})
        for actions, end, label in edges:
            trace = sim.trace(landing, actions)
            self.assertEqual(trace[-1], end, label)
            self.assertTrue(all(goal.achieved(state) for state in trace), label)
            self.assertTrue(end.cassette_blocks[0].collidable, label)
            self.assertEqual(end.dashes, 1, label)
            self.assertFalse(goal.achieved(sim.step(end, actions[-1])), label)

    def test_late_active_phase_waits_for_next_observed_activation(self):
        sim,start,goal=fixture()
        for _ in range(40):
            start=sim.step(start,0)
        self.assertTrue(start.cassette_blocks[0].collidable)
        actions,end,label=next(cassette_support_edges(sim,start,goal,max_frames=300))
        delay=int(label.split(':')[1])
        self.assertGreater(delay,48)
        self.assertEqual(actions[:delay],[0]*delay)
        replay=sim.trace(start,actions)
        self.assertTrue(all(state.on_ground and not state.dead for state in replay[:delay+1]))
        self.assertEqual(replay[-1],end)
        self.assertTrue(goal.achieved(end))
        self.assertGreater(end.cassette_clock.beat_index,start.cassette_clock.beat_index+8)

    def test_unavailable_phase_airborne_and_expired_deadline_produce_no_edge(self):
        sim, start, goal = fixture()
        for invalid in (replace(start, dead=True), replace(start, on_ground=False),
                        replace(start, cassette_clock=None)):
            self.assertEqual(list(cassette_support_edges(sim, invalid, goal)), [])
        self.assertEqual(list(cassette_support_edges(sim, start, goal, deadline=0)), [])
        self.assertEqual(list(cassette_launch_edges(sim, start, goal)), [])
        missing = replace(start, cassette_clock=replace(start.cassette_clock, active=False),
                          cassette_blocks=(replace(start.cassette_blocks[0],
                                                   activated=False, collidable=False),))
        self.assertEqual(list(cassette_support_edges(sim, missing, goal)), [])


if __name__ == "__main__":
    unittest.main()
