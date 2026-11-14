"""A short real public suffix gets a turn before unrelated private contacts."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import (advance_route_progress, normalize_route,
                                         route_complete, start_route_progress)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_planning import search_mechanism_routes
from celeste_search.mechanism_moves import active_dash_block_edges
from celeste_search.mechanism_routes import dash_break_goals
from celeste_search.physics import DASH, JUMP, LEFT, RIGHT, Simulator
from celeste_search.planner import SearchResult, search_leg


def fixture(points=None, *, limits=None):
    room = Room("suffix", 0, 0, 256, 160, ("0"*32,)*18+("1"*32,)*2,
                (), ((16., 144.),))
    sim = Simulator(MapData("near-public", {room.name:room}, Path("near-public.bin")))
    start = replace(sim.initial_state(), frame=121, rem_x=.25, rem_y=-.125)
    # Derive the airborne root through real movement, retaining held input,
    # upward speed and subpixel position rather than inventing a landing.
    for mask in [RIGHT|JUMP]*10:
        start = sim.step(start, mask)
    points = points or [dict(room=room.name, x=64, y=136),
                        dict(room=room.name, x=96, y=136)]
    route = normalize_route(sim.map, dict(waypoints=points,
        segments=limits or [{} for _ in points]))
    goals = [_selected_goal(point)[0] for point in route["waypoints"]]
    return sim, start, route, goals


class MechanismNearPublicTests(unittest.TestCase):
    def test_real_airborne_two_point_suffix_runs_before_private_contacts(self):
        sim, start, route, goals = fixture()
        calls = []

        def actual_search(wrapped, state, target, **kwargs):
            calls.append((state, target, kwargs["max_expansions"], wrapped.no_dash))
            return search_leg(wrapped, state, target, **kwargs)

        with patch("celeste_search.mechanism_planning.search_leg", side_effect=actual_search), \
             patch("celeste_search.mechanism_routes.relay_candidates",
                   side_effect=AssertionError("Public suffix must precede private contact inventory")):
            result = search_mechanism_routes(sim, start, route, goals,
                deadline=time.monotonic()+2, max_expansions=192, max_frames=160, use_detours=False)
        self.assertTrue(result[2], result[6])
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][1], goals[0])
        self.assertIs(calls[1][1], goals[1])
        self.assertTrue(all(no_dash and 0 < quota <= 96 for _,_,quota,no_dash in calls))
        self.assertEqual(result[6]["stages"], ["public:near_no_dash", "public:near_no_dash"])
        state, phase = start, start_route_progress(route, start)
        for mask in result[0]:
            self.assertFalse(mask & DASH)
            before, state = state, sim.step(state, mask)
            phase = advance_route_progress(route, phase, before, state, mask)
            self.assertFalse(state.dead)
            self.assertIsNone(phase.violation)
        self.assertTrue(route_complete(route, phase))
        self.assertEqual(phase, result[5])
        self.assertEqual(state, result[1])
        self.assertLessEqual(result[3], 192)

    def test_returned_inputs_cannot_bypass_no_dash_or_no_ground(self):
        for limits, actions in ((dict(no_dash=True), [RIGHT|DASH]*6+[RIGHT]*45),
                                (dict(no_ground=True), [0]*40+[RIGHT]*45)):
            with self.subTest(limits=limits):
                sim, start, route, goals = fixture([dict(room="suffix", x=80, y=136)], limits=[limits])
                calls = []

                def invalid_suffix(wrapped, state, target, **kwargs):
                    calls.append(wrapped.no_dash)
                    constrained = state
                    for mask in actions:
                        constrained = wrapped.step(constrained, mask)
                        if constrained.dead:
                            break
                    self.assertTrue(constrained.dead)
                    # Even a reported success with a real but forbidden input
                    # sequence cannot bypass the parent's whole-route replay.
                    actual = state
                    for mask in actions:
                        actual = sim.step(actual, mask)
                    return SearchResult(actions, actual, True, 1, len(actions), "claimed_success")

                with patch("celeste_search.mechanism_planning.search_leg", side_effect=invalid_suffix), \
                     patch("celeste_search.mechanism_routes.relay_candidates", return_value=[]):
                    result = search_mechanism_routes(sim, start, route, goals,
                        deadline=time.monotonic()+1, max_expansions=1, max_frames=160, use_detours=False)
                self.assertEqual(calls, [True])
                self.assertFalse(result[2])
                self.assertFalse(route_complete(route, result[5]))
                self.assertEqual(result[0], [])
                self.assertEqual(result[1], start)
                self.assertEqual(result[3], 1)

    def test_small_total_budget_and_far_root_keep_original_limits(self):
        for target_x, is_near in ((80, True), (232, False)):
            with self.subTest(target_x=target_x):
                sim, start, route, goals = fixture([dict(room="suffix", x=target_x, y=136)])
                calls = []

                def spend_budget(wrapped, state, target, **kwargs):
                    calls.append((target is goals[0], kwargs["max_expansions"], kwargs["max_frames"], wrapped.no_dash))
                    return SearchResult([], state, False, kwargs["max_expansions"], 0, "budget")

                with patch("celeste_search.mechanism_planning.search_leg", side_effect=spend_budget), \
                     patch("celeste_search.mechanism_routes.relay_candidates", return_value=[]):
                    result = search_mechanism_routes(sim, start, route, goals,
                        deadline=time.monotonic()+1, max_expansions=7, max_frames=17, use_detours=False)
                self.assertEqual(calls, [(is_near, 7, 17, True)])
                self.assertEqual(result[3], 7)
                self.assertFalse(result[2])

    def test_new_opened_world_gets_a_suffix_before_its_high_cost_is_ranked(self):
        room = Room("opened", 0, 0, 352, 192, ("0"*44,)*22+("1"*44,)*2,
                    (Entity("dashBlock", 7, 200, 128, 16, 48),), ((128.,176.),))
        sim = Simulator(MapData("opened-cost", {room.name:room}, Path("opened-cost.bin")))
        start = replace(sim.initial_state(), frame=63, rem_x=.125)
        route = normalize_route(sim.map, dict(room=room.name, x=248, y=168))
        public = _selected_goal(route["waypoints"][0])[0]
        private = next(goal for goal in dash_break_goals(sim, start) if goal.dash_dx == 1)
        ranked, probes = [], []

        def expensive_real_break(wrapped, state, target, **kwargs):
            # Two genuine ground dashes spend frames and raise path cost but
            # restore their charges only through actual settling. This makes
            # the eventual opening expensive, without fabricating resources.
            current, prefix = state, []
            for direction in (LEFT, RIGHT):
                for mask in [direction|DASH]*6+[0]*60:
                    current = wrapped.step(current, mask)
                    self.assertFalse(current.dead)
                    prefix.append(mask)
            self.assertTrue(current.on_ground)
            self.assertEqual(current.dashes, 1)
            kwargs["stats"]["expansions"] += 2
            kwargs["stats"]["simulated_frames"] += len(prefix)
            kwargs["max_frames"] -= len(prefix)
            return [(prefix+actions, final, label) for actions, final, label in
                    active_dash_block_edges(wrapped, current, target, **kwargs)]

        def rank(wrapped, state, target):
            ranked.append(state)
            return 10000. if state.broken else 0.

        def actual_suffix(wrapped, state, target, **kwargs):
            probes.append(state)
            if state.broken:
                self.assertFalse(any(candidate.broken for candidate in ranked),
                                 "The opened state must be probed before cost-based queue ranking")
            return search_leg(wrapped, state, target, **kwargs)

        with patch("celeste_search.mechanism_moves.active_dash_block_edges", side_effect=expensive_real_break), \
             patch("celeste_search.mechanism_routes.relay_candidates", return_value=[private]), \
             patch("celeste_search.mechanism_routes.relay_priority", side_effect=rank), \
             patch("celeste_search.mechanism_planning.search_leg", side_effect=actual_suffix):
            result = search_mechanism_routes(sim, start, route, [public],
                deadline=time.monotonic()+2, max_expansions=400, max_frames=600, use_detours=False)
        self.assertTrue(result[2], result[6])
        self.assertTrue(any(state.broken for state in probes))
        self.assertTrue(any(row["trigger"] == "opened_world" and row["dashes_used"] >= 3
                            for row in result[6]["near_public_probes"]))
        actual, phase, dash_starts = start, start_route_progress(route, start), 0
        for mask in result[0]:
            before, actual = actual, sim.step(actual, mask)
            dash_starts += bool(sim.last_step_started_dash)
            phase = advance_route_progress(route, phase, before, actual, mask)
            self.assertFalse(actual.dead)
            self.assertIsNone(phase.violation)
        self.assertGreaterEqual(dash_starts, 3)
        self.assertTrue(route_complete(route, phase))
        self.assertEqual(actual, result[1])
        self.assertEqual(actual.broken, frozenset({"opened:7"}))


if __name__ == "__main__":
    unittest.main()
