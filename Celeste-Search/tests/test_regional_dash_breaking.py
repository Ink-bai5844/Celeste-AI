"""Local block actions continue from actual regional arrivals, not teleports."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete, RouteProgress
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import active_dash_block_edges
from celeste_search.physics import RIGHT, Simulator
from celeste_search.planner import SearchResult
from celeste_search.regional_planning import (_Arrival, _diverse_arrivals,
                                              _nearby_breaks, search_regional_routes)


def fixture(*, mirror=False, two=True, can_dash=True):
    ox, oy = (-704, -1200) if mirror else (0, 0)
    px = lambda x: ox+(320-x if mirror else x)
    blocks = [Entity("dashBlock", 7, px(104) if mirror else px(88), oy+128,
                     16, 48, attrs={"canDash": can_dash})]
    if two:
        blocks.append(Entity("dashBlock", 8, px(200) if mirror else px(184), oy+128, 16, 48))
    room = Room("local", ox, oy, 320, 192, ("0"*40,)*22+("1"*40,)*2,
                tuple(blocks), ((px(16), oy+176.),))
    sim = Simulator(MapData("local-break", {"local": room}, Path("local-break.bin")))
    start = replace(sim.initial_state(), frame=231, rem_x=.125, rem_y=-.0625)
    route = normalize_route(sim.map, dict(room="local", x=px(272), y=oy+168))
    return sim, start, route


class RegionalDashBreakingTests(unittest.TestCase):
    def solve(self, *, mirror=False, no_dash=False, no_ground=False,
              can_dash=True, fake=False, unreachable=False, seconds=4.):
        sim, start, route = fixture(mirror=mirror, can_dash=can_dash,
                                    two=can_dash and not fake and not unreachable)
        route["segments"][0].update(no_dash=no_dash, no_ground=no_ground)
        if unreachable:
            route["waypoints"][0]["y"] = sim.map.rooms["local"].y+8
        goal = _selected_goal(route["waypoints"][0])[0]
        calls, probes = [], []

        def local_action(wrapped, origin, target, **kwargs):
            calls.append((origin, target, kwargs["max_expansions"]-kwargs["stats"]["expansions"]))
            if fake:
                return [([0], replace(origin, broken=origin.broken|{target.identity}), "claimed_break")]
            return active_dash_block_edges(wrapped, origin, target, **kwargs)

        def ordinary_probe(wrapped, origin, target, **kwargs):
            # Disable the general graph policy so only the new deliberate
            # actions can open these walls. Every walk/jump/break remains raw
            # physics, including the entire exact public suffix afterwards.
            probes.append(origin)
            return SearchResult([], origin, False, 1, 0, "controlled_no_path")

        with patch("celeste_search.regional_planning.search_leg", side_effect=ordinary_probe), \
             patch("celeste_search.regional_planning.active_dash_block_edges", side_effect=local_action):
            result = search_regional_routes(sim, start, route, [goal],
                deadline=time.monotonic()+seconds, max_expansions=1000, max_frames=600)
        return sim, start, route, result, calls, probes

    def test_two_local_breaks_rebound_continue_and_exact_public_suffix(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                sim, start, route, result, calls, probes = self.solve(mirror=mirror)
                self.assertTrue(result[2], result[6])
                self.assertTrue(route_complete(route, result[5]))
                self.assertEqual(result[1].broken, frozenset({"local:7", "local:8"}))
                self.assertTrue(any(state.broken and not state.on_ground for state in probes),
                                "The genuine airborne rebound must enter the regional frontier")
                self.assertTrue(any("local:7" in state.broken and target.identity == "local:8"
                                    for state, target, _ in calls))
                self.assertTrue(all(0 < quota <= 32 for _, _, quota in calls))
                self.assertTrue(any(state.frame > start.frame for state, _, _ in calls),
                                "The launch must include a real approach from the original state")
                state = start
                for mask in result[0]:
                    state = sim.step(state, mask)
                    self.assertFalse(state.dead)
                self.assertEqual(state, result[1])
                self.assertFalse(start.broken)
                self.assertGreaterEqual(sum("active_dash_block" in stage
                                            for stage in result[6]["stages"]), 2)

    def test_constraints_identity_and_private_destruction_cannot_fake_success(self):
        for options in (dict(no_dash=True), dict(no_ground=True), dict(can_dash=False),
                        dict(fake=True), dict(unreachable=True)):
            with self.subTest(options=options):
                _, start, route, result, calls, _ = self.solve(**options)
                self.assertFalse(result[2])
                self.assertFalse(route_complete(route, result[5]))
                self.assertFalse(start.broken)
                if options.get("no_dash") or options.get("can_dash") is False:
                    self.assertEqual(calls, [])
                if options.get("fake"):
                    self.assertTrue(calls)
                    self.assertEqual(result[6]["active_breaks"]["accepted"], 0)
                if options.get("unreachable"):
                    self.assertGreater(result[6]["active_breaks"]["accepted"], 0)

    def test_discovery_stays_local_and_reserves_distinct_block_identities(self):
        sim, start, _ = fixture()
        origin = _Arrival(start, [], RouteProgress())
        # The first face is 64px away; the second cannot consume this node's
        # local action quota. No simulation or resource changes for discovery.
        with patch.object(sim, "step", side_effect=AssertionError("geometry only")):
            nearby = _nearby_breaks(sim, [origin])
        self.assertEqual([target.identity for _, target in nearby], ["local:7"])
        self.assertTrue(all(seed is origin for seed, _ in nearby))
        far = _Arrival(replace(start, x=start.x-8), [], RouteProgress())
        self.assertEqual(_nearby_breaks(sim, [far]), [])
        # Inputs here are discovery-only representatives; execution tests
        # above certify real prefixes independently from the original state.
        between = _Arrival(replace(start, x=144), [], RouteProgress())
        near_both = _nearby_breaks(sim, [between])
        self.assertEqual({target.identity for _, target in near_both}, {"local:7", "local:8"})
        self.assertLessEqual(len(near_both), 2)

    def test_opened_world_survives_crowded_same_support_selection(self):
        _, start, route = fixture()
        goal = _selected_goal(route["waypoints"][0])[0]
        ordinary = [_Arrival(replace(start, x=150+i*2, rem_x=i*.01), [0]*i, RouteProgress())
                    for i in range(20)]
        opened = _Arrival(replace(start, x=32, broken=frozenset({"local:7"})), [0]*80, RouteProgress())
        retained = _diverse_arrivals(ordinary+[opened], goal, 4)
        self.assertEqual(len(retained), 4)
        self.assertIn(opened, retained,
                      "Geometric proximity must not erase a real opened-world continuation")
        self.assertIs(retained[0], ordinary[0])

    def test_computed_public_hit_is_certified_when_last_local_slice_ends(self):
        sim, start, route = fixture(two=False)
        goal = _selected_goal(route["waypoints"][0])[0]
        clock, called = [10.], []

        def computed_finish(wrapped, origin, target, **kwargs):
            rows = active_dash_block_edges(wrapped, origin, target, **kwargs)
            self.assertTrue(rows)
            actions, current, _ = rows[0]
            actions = list(actions)
            while not goal.achieved(current) and len(actions) < kwargs["max_frames"]:
                current = wrapped.step(current, RIGHT)
                actions.append(RIGHT)
                self.assertFalse(current.dead)
            self.assertTrue(goal.achieved(current))
            called.append(current)
            clock[0] = 12.
            return [(actions, current, "already_computed_public_hit")]

        def unavailable_graph(wrapped, origin, target, **kwargs):
            return SearchResult([], origin, False, 1, 0, "controlled_no_path")

        with patch("celeste_search.regional_planning.time.monotonic", side_effect=lambda: clock[0]), \
             patch("celeste_search.regional_planning.search_leg", side_effect=unavailable_graph), \
             patch("celeste_search.regional_planning.active_dash_block_edges", side_effect=computed_finish):
            result = search_regional_routes(sim, start, route, [goal],
                deadline=12., max_expansions=1000, max_frames=600)
        self.assertEqual(len(called), 1, "No new local search starts after the global deadline")
        self.assertTrue(result[2], result[6])
        self.assertTrue(route_complete(route, result[5]))
        replayed = start
        for mask in result[0]:
            replayed = sim.step(replayed, mask)
        self.assertEqual(replayed, result[1])


if __name__ == "__main__":
    unittest.main()
