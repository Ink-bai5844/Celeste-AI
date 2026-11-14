"""Real refill contacts keep the resource and momentum needed by a relay."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.corridor_routes import RefillPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.refill_moves import refill_approach_edges


def relay_fixture(mirror=False, offset=(0, 0), name="relay"):
    ox, oy = offset
    def x(value):
        return ox + (320 - value if mirror else value)
    entities = tuple(Entity("refill", i, x(px), oy + py, 0, 0)
                     for i, px, py in ((1, 240, 72), (2, 176, 56), (3, 104, 48)))
    room = Room(name, ox, oy, 320, 208, ("0" * 40,) * 20 + ("1" * 40,) * 6,
                entities, ())
    sim = Simulator(MapData("relay", {name: room}, Path("fixture.bin")))
    start = SimState(name, x(286), oy + 160, on_ground=True, grace=6)
    goals = [RefillPortalGoal("refill", name, e.x, e.y + 5.5,
              identity=f"{name}:{e.id}", pickup_rect=(e.x-8,e.y-8,e.x+8,e.y+8),
              role="refill") for e in entities]
    return sim, start, goals


class RefillMovesTests(unittest.TestCase):
    def assert_replay(self, sim, origin, goal, edge):
        actions, expected, label = edge
        state, events = origin, 0
        for mask in actions:
            state = sim.step(state, mask)
            events += bool(sim.last_step_started_dash)
            self.assertFalse(state.dead)
        self.assertEqual(state, expected)
        self.assertTrue(goal.achieved(state))
        self.assertTrue(label.endswith(f"dash_starts={events}"))
        return state

    def test_airborne_chain_replays_from_each_previous_pickup_and_mirrors(self):
        for mirror, offset, name in ((False, (0,0), "relay"),
                                     (True, (-640,-912), "translated")):
            sim, state, goals = relay_fixture(mirror, offset, name)
            for goal in goals:
                before = state
                edges = refill_approach_edges(sim, state, goal,
                    deadline=time.monotonic()+2, max_edges=6)
                self.assertTrue(edges, (mirror, goal.identity))
                self.assertLessEqual(len(edges), 6)
                for edge in edges:
                    self.assert_replay(sim, before, goal, edge)
                self.assertGreater(len({(s.y, round(s.vy/30)) for _,s,_ in edges}), 1)
                state = edges[0][1]
                self.assertEqual(state.dashes, 1)
                self.assertIn(goal.identity, dict(state.refill_timers))

    def test_charge_restored_in_same_leg_still_counts_real_dash(self):
        sim, start, goals = relay_fixture()
        edges = refill_approach_edges(sim, start, goals[0], deadline=time.monotonic()+2)
        self.assertTrue(edges)
        for edge in edges:
            final = self.assert_replay(sim, start, goals[0], edge)
            self.assertEqual((start.dashes, final.dashes), (1,1))
            self.assertTrue(edge[2].endswith("dash_starts=1"))

    def test_falling_pickup_needs_no_dash_and_wrong_identity_cannot_pass(self):
        sim, _, goals = relay_fixture()
        goal = goals[0]
        start = SimState("relay", goal.x, goal.y-40, dashes=0, vy=60)
        edges = refill_approach_edges(sim, start, goal, deadline=time.monotonic()+1)
        self.assertTrue(edges)
        self.assertTrue(all(not any(mask & DASH for mask in actions) for actions,_,_ in edges))
        for edge in edges:
            self.assert_replay(sim, start, goal, edge)
        missing = replace(goal, identity="relay:unknown")
        self.assertEqual(refill_approach_edges(sim, start, missing,
            deadline=time.monotonic()+1), [])

    def test_strict_predicate_does_not_accept_an_old_pickup_from_far_away(self):
        sim, start, goals = relay_fixture()
        start = replace(start, refill_timers=((goals[0].identity,100),))
        self.assertFalse(goals[0].achieved(start))
        self.assertEqual(refill_approach_edges(sim, start, goals[0],
            deadline=time.monotonic()+1, max_frames=1), [])

    def test_wrapper_constraints_are_checked_on_every_simulated_frame(self):
        sim, start, goals = relay_fixture()
        class Constrained:
            no_dash = True
            def __getattr__(self, name):
                return getattr(sim, name)
            def step(self, state, mask):
                result = sim.step(state, mask)
                return replace(result, dead=True, reason="test-hard-limit")
        stats = {}
        self.assertEqual(refill_approach_edges(Constrained(), start, goals[0],
            deadline=time.monotonic()+1, stats=stats), [])
        self.assertEqual(stats["expansions"], stats["simulated_frames"])
        self.assertLessEqual(stats["expansions"], 20)

    def test_deadline_and_cumulative_expansion_budget_are_hard_limits(self):
        sim, start, goals = relay_fixture()
        with patch.object(sim, "step", wraps=sim.step) as step:
            self.assertEqual(refill_approach_edges(sim, start, goals[0],
                deadline=time.monotonic()-1), [])
            self.assertEqual(refill_approach_edges(sim, start, goals[0],
                deadline=time.monotonic()+1, stats={"expansions":4}, max_expansions=4), [])
            step.assert_not_called()
        stats = {"expansions":3}
        refill_approach_edges(sim, start, goals[0], deadline=time.monotonic()+1,
                              stats=stats, max_expansions=4, max_frames=5)
        self.assertEqual(stats["expansions"], 4)
        self.assertEqual(stats["simulated_frames"], 5)
        clock = [0.0]
        original_step = sim.step
        def timed_step(state, mask):
            clock[0] += .6
            return original_step(state, mask)
        with patch("celeste_search.refill_moves.time.monotonic", side_effect=lambda:clock[0]):
            with patch.object(sim, "step", side_effect=timed_step) as step:
                self.assertEqual(refill_approach_edges(sim, start, goals[0], deadline=1), [])
                self.assertEqual(step.call_count, 2)


if __name__ == "__main__":
    unittest.main()
