"""Public waypoint boundaries preserve physical arrivals and segment limits."""
from copy import deepcopy
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import (
    RouteProgress, advance_route_progress, route_complete, start_route_progress,
)
from celeste_search.corridor_planning import _ConstrainedSimulator, search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import DASH, JUMP, LEFT, RIGHT, SimState
from celeste_search.planner import SearchResult


def public_route(*, first=None, second=None):
    default = dict(no_dash=False, no_ground=False)
    return dict(
        waypoints=[dict(room="a", x=16, y=0, tile_size=8),
                   dict(room="a", x=80, y=0, tile_size=8)],
        segments=[dict(default, **(first or {})), dict(default, **(second or {}))],
    )


def direct_corridor(simulator, start, goal, **kwargs):
    return [CorridorRoute("same-geometric-chain", "a", (),
                          ((start.x, start.y), (goal.x, goal.y)), (), 84.)]


def result_of(simulator, start, goal, actions):
    state = start
    for mask in actions:
        state = simulator.step(state, mask)
    return SearchResult(actions, state, bool(actions and goal.achieved(state)),
                        1, len(actions), "deterministic_fixture")


class ArrivalSimulator:
    """Two equal-position arrivals; only one retains the needed next-leg state."""
    map_data = None
    last_step_started_dash = False

    def __init__(self, differing_field):
        self.differing_field = differing_field

    def step(self, state, mask):
        base = dict(frame=state.frame + 1, prev_input=mask, stamina=state.stamina - 2.)
        self.last_step_started_dash = bool(mask & DASH)
        if state.x == 0 and mask in (LEFT, RIGHT):
            arrival = dict(x=20., vx=5., dashes=1)
            if mask == LEFT:
                arrival[self.differing_field] = -5. if self.differing_field == "vx" else 0
            return replace(state, **base, **arrival)
        useful = state.vx > 0 if self.differing_field == "vx" else state.dashes == 1
        if state.x == 20 and mask == RIGHT and useful:
            return replace(state, **base, x=84.)
        # Neither a hidden neutral settling frame nor a resource reset is legal.
        return replace(state, **base, dead=True, reason="fixture_no_edge")


class OrderedCorridorTests(unittest.TestCase):
    def setUp(self):
        self.start = SimState("a", 0., 8., frame=91, stamina=83., dashes=1,
                              rem_x=.125, rem_y=-.25, on_ground=False)

    def solve(self, simulator, route, policy, **limits):
        goals = [_selected_goal(point)[0] for point in route["waypoints"]]
        with patch("celeste_search.corridor_planning.generate_corridor_routes",
                   side_effect=direct_corridor), \
             patch("celeste_search.corridor_planning.search_leg", side_effect=policy):
            return search_corridor_routes(simulator, self.start, route, goals,
                deadline=time.monotonic() + 2, max_expansions=200,
                max_frames=limits.get("max_frames", 12))

    def assert_public_replay(self, simulator, route, actions, final, indices):
        state = self.start
        progress = start_route_progress(route, state)
        actual_indices = []
        for index, mask in enumerate(actions, 1):
            before, prior = state, progress.index
            state = simulator.step(before, mask)
            progress = advance_route_progress(route, progress, before, state, mask)
            actual_indices.extend([index] * (progress.index - prior))
            self.assertIsNone(progress.violation)
        self.assertEqual(state, final)
        self.assertEqual(actual_indices, indices)
        self.assertTrue(route_complete(route, progress))

    def assert_arrival_backtracking(self, field):
        simulator = ArrivalSimulator(field)
        route = public_route()
        original = deepcopy(route)
        suffix_origins = []

        def policy(sim, start, goal, **kwargs):
            if start.x == 0:
                return result_of(sim, start, goal, [LEFT if sim.no_dash else RIGHT])
            suffix_origins.append(start)
            return result_of(sim, start, goal, [RIGHT])

        actions, final, reached, _, _, phase, details = self.solve(simulator, route, policy)
        self.assertTrue(reached)
        self.assertEqual(actions, [RIGHT, RIGHT])
        self.assertEqual(phase.index, 2)
        self.assertGreater(details["backtracks"], 0)
        self.assertEqual({getattr(state, field) for state in suffix_origins},
                         {-5., 5.} if field == "vx" else {0, 1})
        self.assertTrue(all(state.frame == 92 and state.stamina == 81.
                            and state.rem_x == .125 and state.rem_y == -.25
                            for state in suffix_origins))
        self.assertEqual(route, original)
        self.assertEqual((final.frame, final.stamina), (93, 79.))
        self.assert_public_replay(simulator, route, actions, final, [1, 2])

    def test_failed_second_segment_backtracks_first_goal_velocity(self):
        self.assert_arrival_backtracking("vx")

    def test_failed_second_segment_backtracks_first_goal_remaining_dash(self):
        self.assert_arrival_backtracking("dashes")

    def test_first_public_hit_truncates_macro_before_next_segment_is_solved(self):
        class Simulator:
            map_data = None
            last_step_started_dash = False

            def step(self, state, mask):
                self.last_step_started_dash = bool(mask & DASH)
                x = {(0., RIGHT): 20., (20., JUMP): 84.}.get((state.x, mask))
                return replace(state, x=state.x if x is None else x,
                    frame=state.frame + 1, prev_input=mask, dead=x is None)

        simulator = Simulator()
        route = public_route(second=dict(no_dash=True))
        original = deepcopy(route)
        suffix_origins = []

        def policy(sim, start, goal, **kwargs):
            if start.x == 0:
                # A proposed macro continues past the first public hit. The
                # public-route replayer must stop before its invalid dash tail.
                hit = simulator.step(start, RIGHT)
                return SearchResult([RIGHT, DASH], hit, True, 1, 2, "long_macro")
            suffix_origins.append(start)
            return result_of(sim, start, goal, [JUMP])

        value = self.solve(simulator, route, policy)
        self.assertTrue(value[2])
        self.assertEqual(value[0], [RIGHT, JUMP])
        self.assertTrue(suffix_origins)
        self.assertTrue(all(state == simulator.step(self.start, RIGHT)
                            for state in suffix_origins))
        self.assertEqual(route, original)
        self.assert_public_replay(simulator, route, value[0], value[1], [1, 2])

    def test_first_public_point_alone_never_reports_a_complete_route(self):
        simulator = ArrivalSimulator("dashes")
        route = public_route()

        def policy(sim, start, goal, **kwargs):
            if start.x == 0:
                return result_of(sim, start, goal, [RIGHT])
            return SearchResult([], start, False, 1, 0, "suffix_exhausted")

        value = self.solve(simulator, route, policy)
        self.assertFalse(value[2])
        self.assertFalse(route_complete(route, value[5]))
        self.assertLess(value[5].index, 2)
        self.assertTrue(all(row["status"] != "verified"
                            for row in value[6]["topology"].get("corridors", [])))

    def test_needed_neutral_update_is_retained_as_an_action_and_advances_state(self):
        class Simulator(ArrivalSimulator):
            def step(self, state, mask):
                if state.x == 20 and mask == 0:
                    return replace(state, frame=state.frame + 1, prev_input=0,
                                   dashes=1, stamina=state.stamina - 2., rem_y=.375)
                return super().step(state, mask)

        simulator = Simulator("dashes")
        route = public_route()

        def policy(sim, start, goal, **kwargs):
            if start.x == 0:
                return result_of(sim, start, goal, [LEFT])
            return result_of(sim, start, goal, [0, RIGHT])

        value = self.solve(simulator, route, policy)
        self.assertTrue(value[2])
        self.assertEqual(value[0], [LEFT, 0, RIGHT])
        self.assertEqual((value[1].frame, value[1].stamina, value[1].rem_y), (94, 77., .375))
        self.assert_public_replay(simulator, route, value[0], value[1], [1, 3])

    def assert_segment_constraint(self, flag, first, second, expected):
        class Simulator:
            map_data = None
            last_step_started_dash = False

            def step(self, state, mask):
                self.last_step_started_dash = bool(mask & DASH)
                return replace(state, x=20. if state.x == 0 else 84.,
                    frame=state.frame + 1, prev_input=mask,
                    on_ground=(flag == "no_ground" and state.x == 20),
                    state=2 if mask & DASH else 0,
                    dashes=0 if mask & DASH else state.dashes)

        simulator = Simulator()
        route = public_route(first={flag: first}, second={flag: second})

        def policy(sim, start, goal, **kwargs):
            action = RIGHT if start.x == 0 else DASH if flag == "no_dash" else JUMP
            return result_of(sim, start, goal, [action])

        value = self.solve(simulator, route, policy)
        self.assertEqual(value[2], expected)
        if expected:
            self.assertEqual(value[5].index, 2)
            self.assert_public_replay(simulator, route, value[0], value[1], [1, 2])
        else:
            self.assertFalse(route_complete(route, value[5]))

    def test_no_dash_uses_each_original_segment_without_reset(self):
        for first, second, reached in ((True, False, True), (False, True, False)):
            with self.subTest(first=first, second=second):
                self.assert_segment_constraint("no_dash", first, second, reached)

    def test_no_ground_uses_each_original_segment_without_reset(self):
        for first, second, reached in ((True, False, True), (False, True, False)):
            with self.subTest(first=first, second=second):
                self.assert_segment_constraint("no_ground", first, second, reached)

    def test_constrained_wrapper_retains_actual_phase_and_airborne_history(self):
        class LandingSimulator:
            map_data = None
            def step(self, state, mask):
                return replace(state, frame=state.frame + 1, on_ground=True, vy=0.)

        route = public_route(second=dict(no_ground=True))
        before = replace(self.start, x=40., on_ground=True)
        value = _ConstrainedSimulator(LandingSimulator(), route,
            phase=RouteProgress(index=1, left_ground=True)).step(before, RIGHT)
        self.assertTrue(value.dead)
        self.assertIn("landing_forbidden", value.reason)

    def test_active_dash_cannot_spill_into_next_no_dash_segment(self):
        class DashArrival:
            map_data = None
            last_step_started_dash = True
            def step(self, state, mask):
                return replace(state, x=20., frame=state.frame + 1,
                               state=2, dash_frames=4, dashes=0)

        route = public_route(second=dict(no_dash=True))

        def policy(sim, start, goal, **kwargs):
            return result_of(sim, start, goal, [DASH])

        value = self.solve(DashArrival(), route, policy)
        self.assertFalse(value[2])
        self.assertEqual(value[5].index, 0)


if __name__ == "__main__":
    unittest.main()
