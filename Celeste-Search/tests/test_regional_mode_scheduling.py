"""Ordinary-first region guidance must leave turns for dash-capable probes."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete
from celeste_search.maps import MapData, Room
from celeste_search.physics import Simulator
from celeste_search.planner import SearchResult
from celeste_search.regional_planning import GroundInterval, search_regional_routes


def fixture():
    # Three high platforms are separate real supports, with a launch floor on
    # the left. The public point is deliberately above normal jump reach.
    grid = [list("0"*64) for _ in range(30)]
    for y in range(28, 30):
        grid[y] = list("1"*64)
    for left in (16, 32, 48):
        for y in range(20, 28):
            for x in range(left, left+4):
                grid[y][x] = "1"
    room = Room("modes", 0, 0, 512, 240, tuple("".join(row) for row in grid), (), ((16.,224.),))
    sim = Simulator(MapData("regional-modes", {room.name: room}, Path("regional-modes.bin")))
    route = normalize_route(sim.map, dict(room=room.name, x=480, y=120))
    regions = tuple(GroundInterval(room.name, x+4., x+28., 160.) for x in (128,256,384))
    return sim, route, regions


class RegionalModeSchedulingTests(unittest.TestCase):
    def run_search(self, *, no_dash=False, charge_quota=False, budget=1800):
        sim, route, regions = fixture()
        route["segments"][0]["no_dash"] = no_dash
        start = sim.initial_state()
        public = _selected_goal(route["waypoints"][0])[0]
        calls = []

        def controlled_probe(wrapped, actual, target, **kwargs):
            private = getattr(target, "portal", target)
            region = getattr(private, "region", None)
            calls.append(dict(origin=actual, no_dash=wrapped.no_dash,
                region=region.key if region else None, quota=kwargs["max_expansions"]))
            # Consume only a controlled budget, never invent a reached state
            # or a successful public target. Walk/jump seed generation still
            # simulates the actual map and original movement state.
            cost = kwargs["max_expansions"] if charge_quota else 1
            return SearchResult([], actual, False, cost, 0, "controlled_budget")

        with patch("celeste_search.regional_planning.search_leg", side_effect=controlled_probe), \
             patch("celeste_search.regional_planning.target_ground_intervals", return_value=regions):
            result = search_regional_routes(sim, start, route, [public],
                deadline=time.monotonic()+3, max_expansions=budget, max_frames=300)
        self.assertFalse(result[2])
        self.assertFalse(route_complete(route, result[5]))
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], start)
        self.assertLessEqual(result[3], budget)
        self.assertEqual((calls[0]["region"], calls[0]["no_dash"], calls[0]["quota"]),
                         (None, True, 100))
        return calls, regions, result

    def test_every_region_gets_ordinary_then_relaxed_before_alternate_seeds(self):
        calls, regions, _ = self.run_search()
        region_calls = [call for call in calls if call["region"]]
        first_round = region_calls[:len(regions)*2]
        self.assertEqual([(call["region"], call["no_dash"]) for call in first_round],
                         [(region.key, mode) for region in regions for mode in (True,False)])
        self.assertEqual(len({call["origin"] for call in first_round}), 1,
                         "Other launch positions must not delay the first dash-capable turn")
        self.assertTrue(all(call["quota"] <= 240 for call in region_calls))

    def test_small_region_allowance_is_shared_before_an_ordinary_probe_consumes_it(self):
        calls, regions, result = self.run_search(charge_quota=True, budget=360)
        region_calls = [call for call in calls if call["region"]]
        self.assertEqual([(call["region"], call["no_dash"]) for call in region_calls[:6]],
                         [(region.key, mode) for region in regions for mode in (True,False)])
        self.assertTrue(all(call["quota"] > 0 for call in region_calls[:6]))
        self.assertLessEqual(result[3], 360)

    def test_hard_no_dash_segment_never_schedules_a_relaxed_probe(self):
        calls, regions, _ = self.run_search(no_dash=True)
        self.assertTrue(calls)
        self.assertTrue(all(call["no_dash"] for call in calls))
        self.assertEqual([call["region"] for call in calls if call["region"]][:3],
                         [region.key for region in regions])


if __name__ == "__main__":
    unittest.main()
