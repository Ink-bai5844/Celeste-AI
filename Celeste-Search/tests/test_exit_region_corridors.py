"""Regional exit topology keeps distinct physical approaches and lane bounds."""
from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_targets import expand_exit_target, room_exit_targets, transition_target_reached
from celeste_search.corridor_routes import (
    _ClearanceGrid, clear_corridor_cache, generate_corridor_routes,
)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import RIGHT, SimState, Simulator
from celeste_search.terminal_entry import terminal_entry_certificate


def fixture(*, source_entities=(), destination_entities=(), destination_floor=88):
    source = Room("source", 0, 0, 128, 96, ("0" * 16,) * 11 + ("1" * 16,),
                  source_entities, ((16., 88.),))
    rows = (("0" * 16,) * (destination_floor // 8)
            + ("1" * 16,) * (12 - destination_floor // 8))
    destination = Room("destination", 128, 0, 128, 96, rows,
                       destination_entities, ((160., float(destination_floor)),))
    data = MapData("exit-corridors", {source.name: source, destination.name: destination},
                   Path("exit-corridors.bin"))
    simulator = Simulator(data)
    exact = next(point for point in room_exit_targets(data, source.name)
                 if point["edge"] == "right" and point["y"] == 8)
    return data, simulator, simulator.initial_state("source"), exact


class ExitRegionCorridorTests(unittest.TestCase):
    def setUp(self):
        clear_corridor_cache()

    def assert_clear_geometry(self, simulator, routes):
        grid = _ClearanceGrid(simulator, simulator.map.rooms["source"], 4, time.monotonic() + 2)
        for route in routes:
            self.assertTrue(all(grid.clear(point) for point in route.points))
            self.assertTrue(all(grid.segment_clear(a, b) for a, b in zip(route.points, route.points[1:])))
            self.assertFalse(route.to_dict()["physically_verified"])

    def test_members_produce_distinct_real_geometric_endpoints(self):
        data, simulator, start, exact = fixture()
        goal, _ = _selected_goal(expand_exit_target(data, exact))
        routes = generate_corridor_routes(simulator, start, goal, max_routes=4)
        self.assertEqual(len(routes), 4)
        self.assertEqual(len({route.points[-1] for route in routes}), len(routes))
        self.assertEqual(len({route.id for route in routes}), len(routes))
        self.assertTrue(all(route.points[-1][0] == 124 for route in routes))
        self.assertGreater(max(route.points[-1][1] for route in routes)
                           - min(route.points[-1][1] for route in routes), 40)
        self.assert_clear_geometry(simulator, routes)

    def test_hazard_blocks_clicked_lane_but_another_region_lane_has_real_route_and_crossing(self):
        spike = Entity("spikesLeft", 1, 128, 0, 0, 48)
        data, simulator, start, exact = fixture(source_entities=(spike,))
        exact_goal, _ = _selected_goal(exact)
        self.assertEqual(generate_corridor_routes(simulator, start, exact_goal), [])
        region = expand_exit_target(data, exact)
        goal, _ = _selected_goal(region)
        routes = generate_corridor_routes(simulator, start, goal, max_routes=4)
        self.assertTrue(routes)
        self.assertTrue(all(route.points[-1][1] >= 60 for route in routes))
        self.assert_clear_geometry(simulator, routes)
        state = start
        for _ in range(120):
            state = simulator.step(state, RIGHT)
            self.assertFalse(state.dead)
            if state.room != start.room:
                break
        self.assertTrue(goal.achieved(state))
        self.assertTrue(transition_target_reached(state, region))
        self.assertFalse(transition_target_reached(state, exact))

    def test_partial_lane_approach_uses_clear_bounds_instead_of_blocked_tile_midpoint(self):
        barrier = Entity("solid", 1, 120, 32, 16, 1)
        data, simulator, _, exact = fixture(source_entities=(barrier,))
        region = expand_exit_target(data, exact)
        partial = next(member for member in region["exit_region"] if member["y"] == 24)
        self.assertLess(partial["arrival_bounds"][3], 32)
        # This is a real canonical member selected as an individual geometry
        # proposal, as occurs when a certificate leaves only one useful lane.
        goal, _ = _selected_goal(dict(region, exit_region=[partial]))
        start = SimState("source", 16, 24)
        routes = generate_corridor_routes(simulator, start, goal, max_routes=1)
        self.assertTrue(routes)
        self.assertEqual(routes[0].points[-1], (124, 32))
        self.assert_clear_geometry(simulator, routes)

    def test_single_certified_lane_guides_geometry_and_goal_away_from_clicked_unsafe_lane(self):
        # Only the y=40 lane clears both Lookout interaction envelopes; the
        # destination floor makes that lane's release safe as well.
        viewers = (Entity("towerviewer", 1, 160, 24, 0, 0),
                   Entity("towerviewer", 2, 160, 72, 0, 0))
        data, simulator, _, exact = fixture(destination_entities=viewers, destination_floor=48)
        region = expand_exit_target(data, exact)
        certificate = terminal_entry_certificate(simulator, region, is_terminal=True)
        self.assertIsNotNone(certificate)
        self.assertEqual(len(certificate.lanes), 1)
        self.assertEqual(certificate.lanes[0].tile_y, 40)
        goal, _ = _selected_goal(region)
        goal = replace(goal, terminal_entry=certificate)
        start = SimState("source", 16, 48)
        routes = generate_corridor_routes(simulator, start, goal, max_routes=2)
        self.assertTrue(routes)
        self.assertTrue(all(route.points[-1][1] >= 44 for route in routes))
        self.assert_clear_geometry(simulator, routes)
        allowed = SimState("destination", 133, 46, reason="transition")
        unsafe = replace(allowed, y=18)
        self.assertTrue(goal.achieved(allowed))
        self.assertTrue(transition_target_reached(unsafe, region))
        self.assertFalse(goal.achieved(unsafe))
        self.assertGreater(goal.waypoint(start)[1], 44)

    def test_single_remaining_member_retains_distinct_upper_and_lower_obstacle_routes(self):
        obstacle = Entity("solid", 3, 56, 24, 16, 40)
        data, simulator, _, exact = fixture(source_entities=(obstacle,))
        region = expand_exit_target(data, exact)
        member = next(row for row in region["exit_region"] if row["y"] == 40)
        goal, _ = _selected_goal(dict(region, exit_region=[member]))
        routes = generate_corridor_routes(simulator, SimState("source", 16, 48), goal, max_routes=4)
        self.assertGreaterEqual(len(routes), 2)
        self.assertTrue(any(min(y for _, y in route.points) <= 24 for route in routes))
        self.assertTrue(any(max(y for _, y in route.points) >= 76 for route in routes))
        self.assert_clear_geometry(simulator, routes)


if __name__ == "__main__":
    unittest.main()
