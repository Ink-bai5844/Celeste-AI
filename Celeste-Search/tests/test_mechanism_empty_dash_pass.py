"""Skip duplicate recipes without losing later refills or buffered launches.

The small scripted simulator isolates scheduler/resource contracts; it is not
a replacement for the separate native or physical movement regression tests.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import advance_route_progress, route_complete, start_route_progress
from celeste_search.mechanism_planning import search_mechanism_routes
from celeste_search.physics import DASH, DOWN, GRAB, JUMP, RIGHT, SimState
from celeste_search.planner import Goal, SearchResult
from celeste_search.wall_descents import wall_descent_edges


@dataclass(frozen=True)
class Contact(Goal):
    facing: int = 1

    def achieved(self, state):
        return not state.dead and state.room == self.room and state.x == 20


class ResourceSimulator:
    map_data = None
    _tiles = {}
    last_step_started_dash = False

    def step(self, state, mask):
        self.last_step_started_dash = False
        changed = dict(frame=state.frame+1, prev_input=mask)
        if state.dash_buffer > 0:
            # A refill coincides with the pending dash. No DASH bit is needed
            # on this update; no-dash wrappers must inspect the actual event.
            self.last_step_started_dash = True
            return replace(state, **changed, x=404., y=8., state=2,
                           dashes=0, dash_buffer=0)
        if mask & DOWN and state.x == 0:
            # This fixture's refill is reached only by a real descent update.
            return replace(state, **changed, y=16., dashes=1,
                           refill_timers=(("a:refill", 150),))
        if mask & DASH and state.dashes > 0:
            self.last_step_started_dash = True
            return replace(state, **changed, x=404., y=8., state=2, dashes=0)
        return replace(state, **changed)


class EmptyDashPassTests(unittest.TestCase):
    paths = {
        "mechanism": "celeste_search.mechanism_moves.mechanism_transfer_edges",
        "static": "celeste_search.wall_transfers.static_wall_transfer_edges",
        "underpass": "celeste_search.wall_underpasses.wall_underpass_edges",
        "approach": "celeste_search.wall_moves.wall_approach_edges",
        "descent": "celeste_search.wall_descents.wall_descent_edges",
    }

    def solve(self, *, buffer=0, no_dash=False, producer=None, real_descent=False,
              generic=False):
        simulator = ResourceSimulator()
        start = SimState("a", 0., 8., state=1, facing=1, dashes=0,
                         dash_buffer=buffer, stamina=90., frame=71)
        route = dict(waypoints=[dict(room="a", x=400, y=0, tile_size=8)],
                     segments=[dict(no_dash=no_dash, no_ground=False)])
        original = deepcopy(route)
        public = _selected_goal(route["waypoints"][0])[0]
        private = Contact("wall_grip", "a", 20., 16., id="actual_contact")
        calls = {name: [] for name in self.paths}
        searched = []

        def macro(name):
            def emit(wrapped, origin, target, **kwargs):
                calls[name].append(wrapped.no_dash)
                if name == "descent" and real_descent:
                    return wall_descent_edges(wrapped, origin, target, **kwargs)
                if producer == name:
                    after = wrapped.step(origin, 0)
                    if not after.dead and target.achieved(after):
                        return [([0], after, "buffered_actual_launch")]
                return []
            return emit

        def search(wrapped, origin, target, **kwargs):
            searched.append(wrapped.no_dash)
            if not generic:
                return SearchResult([], origin, False, 1, 0, "fixture_no_contact")
            actions = [DOWN | GRAB | RIGHT, DASH | RIGHT]
            state = origin
            for mask in actions:
                state = wrapped.step(state, mask)
                if state.dead:
                    break
            return SearchResult(actions, state, target.achieved(state), 1, 2,
                                "fixture_resource_recovery")

        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",
                                      return_value=[private]))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority", return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg", side_effect=search))
            stack.enter_context(patch("celeste_search.zip_grips.zip_grip_edges", return_value=[]))
            if real_descent:
                # Keep the production descent/prefix/recipe code; isolate only
                # source-wall geometry in this scheduler/resource fixture.
                stack.enter_context(patch("celeste_search.wall_descents._static_contact",
                                          return_value=(4., 0., 12., 200.)))
                stack.enter_context(patch("celeste_search.wall_descents._launch_heights",
                                          return_value=[16.]))
            for name, path in self.paths.items():
                stack.enter_context(patch(path, side_effect=macro(name)))
            result = search_mechanism_routes(simulator, start, route, [public],
                deadline=time.monotonic()+2, max_expansions=250, max_frames=64,
                use_detours=False)
        self.assertEqual(route, original)
        return simulator, start, route, result, calls, searched

    def assert_replay(self, simulator, start, route, result):
        self.assertTrue(result[2])
        state, progress = start, start_route_progress(route, start)
        events = []
        for index, mask in enumerate(result[0], 1):
            before, state = state, simulator.step(state, mask)
            if simulator.last_step_started_dash:
                events.append(index)
            progress = advance_route_progress(route, progress, before, state, mask)
            self.assertFalse(state.dead)
            self.assertIsNone(progress.violation)
        self.assertTrue(route_complete(route, progress))
        self.assertEqual(state, result[1])
        self.assertEqual(progress, result[5])
        return events

    def test_empty_dash_skips_only_four_equivalent_relaxed_generators(self):
        _, _, _, result, calls, searched = self.solve()
        self.assertFalse(result[2])
        for name in ("mechanism", "static", "underpass", "approach"):
            self.assertEqual(calls[name], [True], name)
        self.assertEqual(calls["descent"], [True, False])
        self.assertEqual(searched, [True, False])

    def test_origin_gated_recipe_lists_are_identical_without_dash_resource(self):
        from celeste_search.mechanism_moves import _transfer_recipes
        from celeste_search.wall_transfers import _recipes as static_recipes
        from celeste_search.wall_underpasses import _recipes as underpass_recipes
        origin = SimState("a", 0., 8., state=1, facing=1, dashes=0, dash_buffer=0)
        target = Contact("wall_grip", "a", 20., 80.)
        for recipes in (_transfer_recipes, static_recipes, underpass_recipes):
            with self.subTest(generator=recipes.__module__):
                conservative = list(recipes(origin, target, False))
                self.assertTrue(conservative)
                self.assertEqual(conservative, list(recipes(origin, target, True)))
                self.assertFalse(any(mask & DASH for sequence, _ in conservative for mask in sequence))

    def test_descent_collects_refill_then_relaxed_launch_still_runs(self):
        sim, start, route, result, calls, _ = self.solve(real_descent=True)
        self.assertEqual(calls["descent"], [True, False])
        self.assertEqual(start.dashes, 0)
        self.assertEqual(result[0][0], DOWN | GRAB | RIGHT)
        recovered = sim.step(start, result[0][0])
        self.assertEqual(recovered.dashes, 1)
        self.assertEqual(recovered.refill_timers, (("a:refill", 150),))
        self.assertEqual(len(self.assert_replay(sim, start, route, result)), 1)
        self.assertTrue(any(mask & DASH for mask in result[0]))
        _, _, _, rejected, limited_calls, _ = self.solve(real_descent=True, no_dash=True)
        self.assertFalse(rejected[2])
        self.assertEqual(limited_calls["descent"], [True])

    def test_generic_resource_recovery_retains_relaxed_fallback(self):
        sim, start, route, result, calls, searched = self.solve(generic=True)
        self.assertEqual(searched, [True, False])
        self.assertEqual(calls["approach"], [True])
        self.assertEqual(result[0], [DOWN | GRAB | RIGHT, DASH | RIGHT])
        self.assertEqual(self.assert_replay(sim, start, route, result), [2])
        _, _, _, rejected, _, searched = self.solve(generic=True, no_dash=True)
        self.assertFalse(rejected[2])
        self.assertEqual(searched, [True])

    def test_buffer_preserves_both_passes_and_implicit_dash_obeys_no_dash(self):
        _, _, _, _, calls, _ = self.solve(buffer=1)
        for name in self.paths:
            self.assertEqual(calls[name], [True, False], name)
        sim, start, route, result, calls, _ = self.solve(buffer=1, producer="mechanism")
        self.assertEqual(calls["mechanism"], [True, False])
        self.assertEqual(result[0], [0])
        self.assertEqual(self.assert_replay(sim, start, route, result), [1])
        _, _, _, rejected, calls, _ = self.solve(buffer=1, producer="mechanism", no_dash=True)
        self.assertFalse(rejected[2])
        self.assertEqual(calls["mechanism"], [True])


if __name__ == "__main__":
    unittest.main()
