"""Mechanism guidance preserves public routes and actual simulated arrivals."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import (
    RouteProgress, advance_route_progress, route_complete, start_route_progress,
)
from celeste_search.mechanism_planning import search_mechanism_routes
from celeste_search.physics import DASH, JUMP, LEFT, RIGHT, SimState
from celeste_search.planner import Goal, SearchResult


def public_route(*, two=True, first=None, second=None):
    points = [dict(room="a", x=16, y=0, tile_size=8),
              dict(room="a", x=80, y=0, tile_size=8)] if two else [
                  dict(room="a", x=80, y=0, tile_size=8)]
    limits = [first, second] if two else [first]
    return dict(waypoints=points, segments=[
        dict({"no_dash": False, "no_ground": False}, **(value or {}))
        for value in limits])


def outcome(simulator, start, goal, actions):
    state = start
    for action in actions:
        state = simulator.step(state, action)
    return SearchResult(list(actions), state, goal.achieved(state),
                        1, len(actions), "fixture")


class MechanismSimulator:
    map_data = None
    last_step_started_dash = False

    def __init__(self):
        self.calls = 0

    def step(self, state, action):
        self.calls += 1
        self.last_step_started_dash = bool(action & DASH)
        base = dict(frame=state.frame+1, prev_input=action,
                    stamina=state.stamina-2.)
        if state.x == 0 and action in (LEFT, RIGHT):
            phase = 1 if action == LEFT else 2
            return replace(state, **base, x=20.,
                mechanisms=(("a:zip", phase, 7, 12., 0., 0.),))
        if state.x == 20 and action == RIGHT and state.mechanisms[0][1] == 2:
            return replace(state, **base, x=84.)
        return replace(state, **base, dead=True, reason="fixture_no_edge")


class OrderedMechanismTests(unittest.TestCase):
    def setUp(self):
        self.start = SimState("a", 0., 8., frame=91, stamina=83.,
            rem_x=.125, rem_y=-.25, dashes=1, on_ground=False,
            mechanisms=(("a:zip", 0, 0, 12., 0., 0.),))

    def solve(self, simulator, route, policy, *, portals=None, edges=None, **limits):
        goals = [_selected_goal(point)[0] for point in route["waypoints"]]
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",
                                      side_effect=portals or (lambda *a, **k: [])))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority",
                                      return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_moves.mechanism_transfer_edges",
                                      side_effect=edges or (lambda *a, **k: [])))
            stack.enter_context(patch("celeste_search.wall_moves.wall_approach_edges",
                                      return_value=[]))
            stack.enter_context(patch("celeste_search.zip_grips.zip_grip_edges", return_value=[]))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg",
                                      side_effect=policy))
            return search_mechanism_routes(simulator, self.start, route, goals,
                deadline=limits.get("deadline", time.monotonic()+2),
                max_expansions=limits.get("max_expansions", 200),
                max_frames=limits.get("max_frames", 12))

    def assert_replay(self, simulator, route, answer, indices):
        actions, final, reached, _, _, phase, _ = answer
        self.assertTrue(reached)
        state, progress = self.start, start_route_progress(route, self.start)
        actual_indices = []
        for frame, action in enumerate(actions, 1):
            before, old_index = state, progress.index
            state = simulator.step(before, action)
            progress = advance_route_progress(route, progress, before, state, action)
            actual_indices.extend([frame]*(progress.index-old_index))
            self.assertFalse(state.dead)
            self.assertIsNone(progress.violation)
        self.assertEqual(actual_indices, indices)
        self.assertEqual(state, final)
        self.assertEqual(progress, phase)
        self.assertTrue(route_complete(route, progress))

    def test_bounded_near_probe_preserves_cheap_contact_alternatives(self):
        simulator = MechanismSimulator()
        route = public_route(two=False)
        distant = Goal("setup", "a", 50., 8., id="optimistic_but_blocked")
        useful = Goal("setup", "a", 20., 8., id="real_contact")
        searched = []

        def policy(sim, start, target, **kwargs):
            searched.append((start.x, target.key, sim.no_dash, kwargs["max_expansions"]))
            return (outcome(sim, start, target, [RIGHT]) if start.x == 20 else
                    SearchResult([], start, False, 1, 0, "blocked"))

        def edges(sim, start, target, **kwargs):
            if start.x == 0 and target.key == useful.key:
                return [([RIGHT], sim.step(start, RIGHT), "actual_contact")]
            return []

        result = self.solve(simulator, route, policy,
            portals=lambda sim, state, *a, **k: [distant, useful] if state.x == 0 else [],
            edges=edges)
        self.assertTrue(result[2])
        public_key = _selected_goal(route["waypoints"][0])[0].key
        # A short ordinary public probe may fail first. The blocked private
        # target must still not consume a broad leg search before the useful
        # physical contact gets its turn and the public suffix succeeds.
        self.assertEqual(searched, [(0., public_key, True, 96),
                                    (20., public_key, True, 96)])
        self.assert_replay(simulator, route, result, [2])

    def test_equal_position_different_mechanism_phase_retains_alternative_arrivals(self):
        simulator = MechanismSimulator()
        route = public_route(two=False)
        original = deepcopy(route)
        portal = Goal("setup", "a", 20., 8., id="zip_contact")
        suffix_origins = []

        def policy(sim, start, target, **kwargs):
            if start.x == 20:
                suffix_origins.append(start)
                return outcome(sim, start, target, [RIGHT])
            return SearchResult([], start, False, 1, 0, "not_direct")

        def edges(sim, start, target, **kwargs):
            return [([action], sim.step(start, action), str(action)) for action in (LEFT, RIGHT)]

        result = self.solve(simulator, route, policy,
            portals=lambda sim, state, *a, **k: [portal] if state.x == 0 else [], edges=edges)
        self.assertEqual(result[0], [RIGHT, RIGHT])
        self.assertEqual({state.mechanisms[0][1] for state in suffix_origins}, {1, 2})
        self.assertTrue(all(state.frame == 92 and state.stamina == 81.
            and state.rem_x == .125 and state.rem_y == -.25 and state.dashes == 1
            for state in suffix_origins))
        self.assertEqual(route, original)
        self.assert_replay(simulator, route, result, [2])

    def test_private_mechanism_contact_never_counts_as_public_exit(self):
        simulator = MechanismSimulator()
        route = public_route(two=False)
        portal = Goal("setup", "a", 20., 8., id="zip_contact")
        result = self.solve(simulator, route,
            lambda sim, start, goal, **k: SearchResult([], start, False, 1, 0, "no_exit"),
            portals=lambda sim, state, *a, **k: [portal] if state.x == 0 else [],
            edges=lambda sim, start, goal, **k: [([RIGHT], sim.step(start, RIGHT), "contact")])
        self.assertFalse(result[2])
        self.assertEqual(result[5].index, 0)
        self.assertFalse(route_complete(route, result[5]))
        self.assertTrue(result[6]["stages"])

    def test_returned_edge_state_cannot_fabricate_contact(self):
        simulator = MechanismSimulator()
        route = public_route(two=False)
        portal = Goal("setup", "a", 50., 8., id="unreached_contact")
        invented = replace(self.start, x=50., dashes=2, stamina=110.)
        result = self.solve(simulator, route,
            lambda sim, start, goal, **k: SearchResult([], start, False, 1, 0, "no_exit"),
            portals=lambda *a, **k: [portal],
            edges=lambda *a, **k: [([RIGHT], invented, "invented")])
        self.assertFalse(result[2])
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], self.start)

    def test_first_public_hit_truncates_macro_before_next_segment(self):
        class Simulator(MechanismSimulator):
            def step(self, state, action):
                self.last_step_started_dash = bool(action & DASH)
                x = {(0., RIGHT): 20., (20., JUMP): 84.}.get((state.x, action))
                return replace(state, x=state.x if x is None else x,
                    frame=state.frame+1, prev_input=action, dead=x is None)

        simulator, route = Simulator(), public_route(second=dict(no_dash=True))
        original, suffix_origins = deepcopy(route), []

        def policy(sim, start, goal, **kwargs):
            if start.x == 0:
                return SearchResult([RIGHT, DASH], simulator.step(start, RIGHT), True, 1, 2, "long")
            suffix_origins.append(start)
            return outcome(sim, start, goal, [JUMP])

        result = self.solve(simulator, route, policy)
        self.assertEqual(result[0], [RIGHT, JUMP])
        self.assertTrue(all(state == simulator.step(self.start, RIGHT) for state in suffix_origins))
        self.assertEqual(route, original)
        self.assert_replay(simulator, route, result, [1, 2])

    def test_neutral_wait_is_kept_and_advances_mechanism_timer(self):
        class Simulator(MechanismSimulator):
            def step(self, state, action):
                if state.x == 20 and action == 0:
                    self.last_step_started_dash = False
                    return replace(state, frame=state.frame+1, prev_input=0,
                        stamina=state.stamina-2., rem_y=.375,
                        mechanisms=(("a:zip", 2, 6, 12., 0., 0.),))
                return super().step(state, action)

        simulator, route = Simulator(), public_route()
        def policy(sim, start, goal, **kwargs):
            return outcome(sim, start, goal, [LEFT] if start.x == 0 else [0, RIGHT])
        result = self.solve(simulator, route, policy)
        self.assertEqual(result[0], [LEFT, 0, RIGHT])
        self.assertEqual((result[1].frame, result[1].stamina, result[1].rem_y), (94, 77., .375))
        self.assertEqual(result[1].mechanisms[0][2], 6)
        self.assert_replay(simulator, route, result, [1, 3])

    def assert_segment_limit(self, flag, first, second, expected):
        class Simulator(MechanismSimulator):
            def step(self, state, action):
                self.last_step_started_dash = bool(action & DASH)
                return replace(state, x=20. if state.x == 0 else 84.,
                    frame=state.frame+1, prev_input=action,
                    on_ground=flag == "no_ground" and state.x == 20,
                    state=2 if action & DASH else 0,
                    dashes=0 if action & DASH else state.dashes)
        simulator = Simulator()
        route = public_route(first={flag:first}, second={flag:second})
        def policy(sim, start, goal, **kwargs):
            return outcome(sim, start, goal, [DASH if flag == "no_dash" and start.x == 20 else RIGHT])
        result = self.solve(simulator, route, policy)
        self.assertEqual(result[2], expected)
        if expected:
            self.assert_replay(simulator, route, result, [1, 2])
        else:
            self.assertFalse(route_complete(route, result[5]))

    def test_dash_limit_uses_actual_public_segment(self):
        for first, second, expected in ((True,False,True),(False,True,False)):
            with self.subTest(first=first, second=second):
                self.assert_segment_limit("no_dash", first, second, expected)

    def test_ground_limit_uses_actual_public_segment(self):
        for first, second, expected in ((True,False,True),(False,True,False)):
            with self.subTest(first=first, second=second):
                self.assert_segment_limit("no_ground", first, second, expected)

    def test_second_public_point_failure_never_reports_complete(self):
        simulator, route = MechanismSimulator(), public_route()
        def policy(sim, start, goal, **kwargs):
            return (outcome(sim, start, goal, [RIGHT]) if start.x == 0
                    else SearchResult([], start, False, 1, 0, "no_suffix"))
        result = self.solve(simulator, route, policy)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index, 1)
        self.assertFalse(route_complete(route, result[5]))

    def test_exhausted_budget_never_advances_or_returns_success(self):
        for limits in (dict(max_expansions=0), dict(deadline=time.monotonic()-1)):
            with self.subTest(limits=limits):
                simulator = MechanismSimulator()
                result = self.solve(simulator, public_route(),
                    lambda *a, **k: self.fail("search after exhausted budget"), **limits)
                self.assertEqual(simulator.calls, 0)
                self.assertEqual(result[0], [])
                self.assertEqual(result[1], self.start)
                self.assertFalse(result[2])

    def landing_fixture(self, *, rising_contact=False):
        class Simulator(MechanismSimulator):
            def step(self, state, action):
                self.last_step_started_dash = bool(action & DASH)
                x = {(0., RIGHT): 20., (20., RIGHT): 40.,
                     (40., RIGHT): 84., (40., 0): 40.,
                     (84., JUMP): 116.}.get((state.x, action))
                contact = x == 40
                record = state.mechanisms[0]
                return replace(state, x=state.x if x is None else x,
                    frame=state.frame+1, prev_input=action,
                    stamina=state.stamina-2., dead=x is None,
                    on_ground=contact, vy=-1. if contact and rising_contact else 0.,
                    mechanisms=((record[0], record[1], record[2]+1, *record[3:]),))

        setup = Goal("setup", "a", 20., 8., id="actual_contact")
        landing = Goal("surface", "a", 40., 8., id="actual_landing")
        def portals(sim, state, *args, **kwargs):
            return [setup] if state.x == 0 else [landing] if state.x == 20 else []
        def edges(sim, start, goal, **kwargs):
            return [([RIGHT], sim.step(start, RIGHT), "contact")]
        return Simulator(), portals, edges

    def test_landing_suffix_claimed_state_cannot_authorize_a_false_success(self):
        simulator, portals, edges = self.landing_fixture()
        route = public_route(two=False)
        fake_final = replace(self.start, x=84., dashes=2, stamina=110.)
        calls = []
        def suffix(sim, start, original_route, goals, **kwargs):
            calls.append(start)
            # Claimed final/phase are successful, but actual LEFT input dies.
            return [LEFT], fake_final, True, 1, 1, RouteProgress(index=1), {}
        with patch("celeste_search.corridor_planning.search_corridor_routes", side_effect=suffix):
            result = self.solve(simulator, route,
                lambda sim, start, goal, **k: SearchResult([], start, False, 1, 0, "no_exit"),
                portals=portals, edges=edges)
        self.assertTrue(calls)
        self.assertTrue(all(state.x == 40 and state.on_ground for state in calls))
        self.assertFalse(result[2])
        self.assertNotEqual(result[1], fake_final)
        self.assertFalse(route_complete(route, result[5]))

    def test_landing_suffix_replays_actual_phase_and_stops_at_first_public_hit(self):
        simulator, portals, edges = self.landing_fixture()
        route = public_route()
        route["waypoints"].append(dict(room="a", x=112, y=0, tile_size=8))
        route["segments"].append(dict(no_dash=True, no_ground=False))
        original, calls = deepcopy(route), []
        def suffix(sim, start, selected, goals, **kwargs):
            calls.append((start, deepcopy(selected), goals))
            # The final dash belongs to the next public segment and is invalid.
            # Truncation must preserve the first hit then solve that segment.
            return [RIGHT, DASH], replace(start, x=116.), True, 1, 2, RouteProgress(index=2), {}
        def policy(sim, start, goal, **kwargs):
            return (outcome(sim, start, goal, [JUMP]) if start.x == 84
                    else SearchResult([], start, False, 1, 0, "use_mechanisms"))
        with patch("celeste_search.corridor_planning.search_corridor_routes", side_effect=suffix):
            result = self.solve(simulator, route, policy, portals=portals, edges=edges)
        self.assertTrue(calls)
        self.assertTrue(all(selected == dict(waypoints=original["waypoints"][1:],
            segments=original["segments"][1:]) for _, selected, _ in calls))
        self.assertTrue(all(state.dashes == 1 and state.stamina < 83.
                            and state.rem_x == .125 for state, _, _ in calls))
        self.assertNotIn(DASH, result[0])
        self.assertIn(result[0], ([RIGHT, RIGHT, RIGHT, JUMP],
                                  [RIGHT, RIGHT, 0, RIGHT, JUMP]))
        self.assertEqual(route, original)
        self.assert_replay(simulator, route, result, [1, len(result[0])-1, len(result[0])])

    def test_no_ground_segment_does_not_invoke_landing_suffix(self):
        simulator, portals, edges = self.landing_fixture(rising_contact=True)
        route = public_route(two=False, first=dict(no_ground=True))
        def policy(sim, start, goal, **kwargs):
            return (outcome(sim, start, goal, [RIGHT]) if start.x == 40
                    else SearchResult([], start, False, 1, 0, "use_mechanisms"))
        with patch("celeste_search.corridor_planning.search_corridor_routes") as suffix:
            result = self.solve(simulator, route, policy, portals=portals, edges=edges)
        suffix.assert_not_called()
        self.assert_replay(simulator, route, result, [len(result[0])])

    def test_shorter_prefix_is_retained_when_only_it_fits_remaining_frame_budget(self):
        class Simulator(MechanismSimulator):
            def step(self, state, action):
                self.last_step_started_dash = bool(action & DASH)
                x = {(0., LEFT): -1., (-1., 0): -2., (-2., 0): -3.,
                     (-3., RIGHT): 20., (0., DASH): -2.,
                     (20., RIGHT): 40., (40., RIGHT): 60.,
                     (60., RIGHT): 84.}.get((state.x, action))
                # Both branches physically recover their dash at the merge.
                # Their full state then differs only in elapsed frame count.
                dashes = 1 if x == 20 else 0 if action & DASH else state.dashes
                return replace(state, x=state.x if x is None else x,
                    frame=state.frame+1, prev_input=action, dashes=dashes,
                    dead=x is None)

        simulator = Simulator()
        route = public_route(two=False)
        portal = Goal("setup", "a", 20., 8., id="same_recovery_state")
        def edges(sim, start, goal, **kwargs):
            actions = [LEFT, 0, 0, RIGHT] if sim.no_dash else [DASH, 0, RIGHT]
            result = outcome(sim, start, goal, actions)
            return [(actions, result.state, "long_no_dash" if sim.no_dash else "short_dash")]
        def policy(sim, start, goal, **kwargs):
            if start.x != 20:
                return SearchResult([], start, False, 1, 0, "no_direct_path")
            return outcome(sim, start, goal, [RIGHT]*min(3, kwargs["max_frames"]))
        result = self.solve(simulator, route, policy, max_frames=6,
            portals=lambda sim, state, *a, **k: [portal] if state.x == 0 else [], edges=edges)
        self.assertTrue(result[2], "A four-frame zero-dash prefix must not discard the three-frame alternative")
        self.assertEqual(result[0], [DASH, 0, RIGHT, RIGHT, RIGHT, RIGHT])
        self.assert_replay(simulator, route, result, [6])


if __name__ == "__main__":
    unittest.main()
