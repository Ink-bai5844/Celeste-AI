"""Bounded corridor pages retain late topology and resume actual discovery."""
from dataclasses import replace
from itertools import islice
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_targets import expand_exit_target, room_exit_targets
from celeste_search.corridor_routes import (
    CorridorCandidates, CorridorRoute, _ClearanceGrid, _discovery_scenarios, clear_corridor_cache,
    generate_corridor_routes,
)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import Goal


def fixture(entities=(), *, width=336, height=224, name="source"):
    room = Room(name, 0, 0, width, height, ("0" * (width // 8),) * (height // 8),
                tuple(entities), ((16., 112.),))
    data = MapData("coverage", {name: room}, Path("coverage.bin"))
    return Simulator(data), SimState(name, 16, 112), Goal("target", name, width - 16, 112)


def winding(route):
    return tuple(item for item in route.signature if item.startswith(("solid:", "oneway:")))


class CorridorCoverageTests(unittest.TestCase):
    def setUp(self):
        clear_corridor_cache()

    def assert_clear(self, simulator, routes):
        grid = _ClearanceGrid(simulator, simulator.map.rooms["source"], 4, time.monotonic() + 2)
        for route in routes:
            self.assertFalse(route.to_dict()["physically_verified"])
            self.assertTrue(all(grid.clear(point) for point in route.points))
            self.assertTrue(all(grid.segment_clear(a, b) for a, b in zip(route.points, route.points[1:])))

    def test_second_page_contains_a_real_obstacle_topology_absent_from_first_six(self):
        obstacles = tuple(Entity("solid", index, 56 * (index + 1), y, 16, 48)
                          for index, y in enumerate((32, 96, 32, 96)))
        simulator, start, goal = fixture(obstacles)
        first = generate_corridor_routes(simulator, start, goal, max_routes=6)
        self.assertIsInstance(first, CorridorCandidates)
        self.assertEqual(len(first), 6)
        self.assertTrue(first.complete)
        with patch("celeste_search.corridor_routes._generate_corridor_target",
                   side_effect=AssertionError("A later cached page must not regenerate earlier work")):
            second = generate_corridor_routes(simulator, start, goal, max_routes=6,
                                              candidate_offset=first.next_offset)
            all_routes = generate_corridor_routes(simulator, start, goal, max_routes=24)
            smaller = generate_corridor_routes(simulator, start, goal, max_routes=3)
        self.assertTrue(second)
        self.assertTrue({winding(route) for route in second} - {winding(route) for route in first})
        self.assertEqual([route.id for route in first + second], [route.id for route in all_routes])
        self.assertEqual([route.id for route in smaller], [route.id for route in first[:3]])
        self.assertEqual(len({route.id for route in all_routes}), len(all_routes))
        self.assertLessEqual(len(all_routes), 24)
        self.assert_clear(simulator, all_routes)

    def test_grid_build_resumes_at_next_unchecked_cell(self):
        simulator, start, _ = fixture(width=80, height=64)
        room=simulator.map.rooms[start.room]
        clock=[0.]; checked=[]
        original=_ClearanceGrid.clear
        def measured(grid, point):
            checked.append(point);clock[0]+=.001
            return original(grid, point)
        with patch("celeste_search.corridor_routes.time.monotonic", side_effect=lambda:clock[0]), \
                patch.object(_ClearanceGrid,"clear",measured):
            grid=_ClearanceGrid(simulator,room,4,.005)
            self.assertFalse(grid.complete)
            first_cursor=grid._next_cell
            for _ in range(100):
                grid.resume_build(clock[0]+.005)
                if grid.complete:break
        self.assertTrue(grid.complete)
        self.assertGreater(first_cursor,0)
        self.assertEqual(len(checked),grid.nx*grid.ny)
        self.assertEqual(len(set(checked)),len(checked),"Completed cells must never be rebuilt")

    def test_astar_slice_preserves_frontier_instead_of_reexpanding_prefix(self):
        obstacles=(Entity("solid",1,104,0,16,128),)
        simulator,start,goal=fixture(obstacles,width=240,height=176)
        grid=_ClearanceGrid(simulator,simulator.map.rooms[start.room],4,time.monotonic()+2)
        first=grid.nearest((start.x,start.y));last=grid.nearest((goal.x,goal.y))
        clock=[0.];expanded=[];original=grid.neighbors
        def measured(cell):
            expanded.append(cell);clock[0]+=.001
            return original(cell)
        grid.neighbors=measured
        route=[]
        with patch("celeste_search.corridor_routes.time.monotonic",side_effect=lambda:clock[0]):
            for _ in range(500):
                grid.deadline=clock[0]+.005
                route=grid.path(first,last,forbidden=set())
                if route:break
        self.assertTrue(route)
        self.assertEqual(len(expanded),len(set(expanded)),"A* must retain its expanded prefix across slices")
        calls=len(expanded)
        self.assertEqual(grid.path(first,last,forbidden=set()),route)
        self.assertEqual(len(expanded),calls)
        self.assertTrue(all(grid.segment_clear(grid.point(a),grid.point(b)) for a,b in zip(route,route[1:])))

    def test_complete_base_route_is_published_when_optional_detours_use_slice(self):
        simulator,start,goal=fixture(width=160,height=160)
        clock=[0.];original=_ClearanceGrid.path
        def expire_at_detours(grid,first,last,penalties=None,forbidden=frozenset(),**options):
            if penalties:
                clock[0]=2.
                return []
            return original(grid,first,last,penalties,forbidden,**options)
        with patch("celeste_search.corridor_routes.time.monotonic",side_effect=lambda:clock[0]), \
                patch.object(_ClearanceGrid,"path",expire_at_detours):
            page=generate_corridor_routes(simulator,start,goal,deadline=1.)
        self.assertTrue(page,"A complete base route must reach the physical scheduler before later detours finish")
        self.assertFalse(page.complete)
        self.assert_clear(simulator,page)

    def test_required_seventh_falling_gate_is_not_removed_by_nearest_six_cutoff(self):
        gates = tuple(Entity("fallingBlock", index + 1, 40 + 32 * index, 0, 8, 160,
                             dict(climbFall=True)) for index in range(7))
        simulator, _, _ = fixture(gates, width=320, height=160)
        start = SimState("source", 16, 80)
        goal = Goal("target", "source", 300, 80)
        routes = generate_corridor_routes(simulator, start, goal, max_routes=6)
        required = {f"source:{index}" for index in range(1, 8)}
        self.assertTrue(routes)
        self.assertTrue(any(set(route.conditional_on) == required for route in routes))
        for route in routes:
            self.assertFalse(route.to_dict()["physically_verified"])
            self.assertEqual(set(route.conditional_on), required)
            for identity in required:
                roles = [portal.kind for portal in route.portals
                         if getattr(portal, "identity", None) == identity]
                self.assertIn("wall_grip", roles)
                self.assertIn("mechanism_clearance", roles)
                self.assertIn("corridor", roles)

    def test_first_exit_page_spreads_across_middle_of_long_opening(self):
        rows = ("0" * 16,) * 31 + ("1" * 16,)
        rooms = {name: Room(name, index * 128, 0, 128, 256, rows, (),
                            ((index * 128 + 16., 248.),))
                 for index, name in enumerate(("source", "destination"))}
        data = MapData("wide-exit", rooms, Path("wide-exit.bin"))
        simulator = Simulator(data)
        point = next(item for item in room_exit_targets(data, "source")
                     if item["edge"] == "right" and item["y"] == 232)
        goal, _ = _selected_goal(expand_exit_target(data, point))
        routes = generate_corridor_routes(simulator, simulator.initial_state("source"), goal, max_routes=6)
        heights = [route.points[-1][1] for route in routes]
        self.assertEqual(len(routes), 6)
        self.assertTrue(any(96 <= y <= 160 for y in heights), heights)
        self.assertGreater(max(heights) - min(heights), 120)
        self.assert_clear(simulator, routes)

    def test_timed_out_page_resumes_next_scene_instead_of_restarting_same_prefix(self):
        gates = (Entity("fallingBlock", 1, 80, 0, 8, 160, dict(climbFall=True)),
                 Entity("fallingBlock", 2, 176, 0, 8, 160, dict(climbFall=True)))
        simulator, start, goal = fixture(gates, width=320, height=160)
        clock = [0.]
        attempted = []

        def one_slow_job(sim, state, target, **options):
            scenario = options["discovery_scenario"]
            attempted.append(scenario)
            clock[0] += .1  # One task exceeds its tiny caller-provided slice.
            identity = "fixture:" + repr(tuple(sorted(scenario)))
            return [CorridorRoute(identity, "source", (), ((16., 112.), (304., 112.)),
                                  (identity,), 288., tuple(sorted(scenario)))]

        with patch("celeste_search.corridor_routes.time.monotonic", side_effect=lambda: clock[0]), \
                patch("celeste_search.corridor_routes._generate_corridor_target", side_effect=one_slow_job):
            first = generate_corridor_routes(simulator, start, goal, max_routes=1, deadline=.05)
            clear_corridor_cache()
            second = generate_corridor_routes(simulator, start, goal, max_routes=1,
                                              candidate_offset=first.next_offset, deadline=.15,
                                              continuation=first.continuation)
        self.assertFalse(first.complete)
        self.assertFalse(second.complete)
        self.assertEqual(first.reason, "generation_time_budget")
        self.assertEqual(first.next_offset, 1)
        self.assertEqual(second.next_offset, 2)
        self.assertNotEqual(attempted[0], attempted[1], "Scene cursor must advance, not just list offset")
        self.assertNotEqual(first[0].id, second[0].id)
        self.assertIs(first.continuation, second.continuation)

    def test_caller_continuation_survives_lru_eviction_and_clear_without_regenerating_pages(self):
        obstacles = tuple(Entity("solid", index, 56 * (index + 1), y, 16, 48)
                          for index, y in enumerate((32, 96, 32, 96)))
        simulator, start, goal = fixture(obstacles)
        first = generate_corridor_routes(simulator, start, goal, max_routes=6)
        handle = first.continuation
        self.assertIsNotNone(handle)
        self.assertEqual(handle.grids, {}, "Finished discovery must release its private large grids")
        with patch("celeste_search.corridor_routes._generate_corridor_target", return_value=[]):
            for frame in range(1, 9):
                generate_corridor_routes(simulator, replace(start, frame=frame), goal)
        with patch("celeste_search.corridor_routes._generate_corridor_target",
                   side_effect=AssertionError("Caller continuation must bypass an evicted LRU record")):
            second = generate_corridor_routes(simulator, start, goal, max_routes=6,
                candidate_offset=first.next_offset, continuation=handle)
            clear_corridor_cache()
            whole = generate_corridor_routes(simulator, start, goal, max_routes=24, continuation=handle)
        self.assertIs(second.continuation, handle)
        self.assertIs(whole.continuation, handle)
        self.assertEqual(first.discovery_cursor, second.discovery_cursor)
        self.assertEqual([row.id for row in first + second], [row.id for row in whole])

    def test_continuation_rejects_different_map_full_start_goal_or_geometry_configuration(self):
        simulator, start, goal = fixture()
        handle = generate_corridor_routes(simulator, start, goal).continuation
        other, _, _ = fixture()
        cases = ((other, start, goal, {}), (simulator, replace(start, frame=1), goal, {}),
                 (simulator, start, replace(goal, y=96), {}),
                 (simulator, start, goal, dict(cell_size=2)),
                 (simulator, start, goal, dict(max_portals=2)))
        for actual_sim, actual_start, actual_goal, options in cases:
            with self.subTest(options=options, state=actual_start), self.assertRaises(ValueError):
                generate_corridor_routes(actual_sim, actual_start, actual_goal,
                                         continuation=handle, **options)
        with self.assertRaises(ValueError):
            generate_corridor_routes(simulator, start, goal, continuation=object())

    def test_full_state_and_goal_change_invalidate_discovery_but_deadline_is_not_exhaustion(self):
        simulator, start, goal = fixture()
        first = generate_corridor_routes(simulator, start, goal)
        self.assertTrue(first)
        with patch("celeste_search.corridor_routes._generate_corridor_target", return_value=[]) as discover:
            generate_corridor_routes(simulator, replace(start, frame=1), goal)
            generate_corridor_routes(simulator, start, replace(goal, y=96))
        self.assertEqual(discover.call_count, 2)
        expired = generate_corridor_routes(simulator, start, goal, deadline=time.monotonic() - 1)
        self.assertEqual(expired, [])
        self.assertFalse(expired.complete)
        self.assertEqual(expired.reason, "generation_time_budget")
        self.assertEqual(generate_corridor_routes(simulator, start, goal, candidate_offset=24), [])
        for value in (-1, True, 1.5):
            with self.subTest(offset=value), self.assertRaises(ValueError):
                generate_corridor_routes(simulator, start, goal, candidate_offset=value)

    def test_large_mechanism_inventory_is_lazy_and_keeps_every_single_gate(self):
        simulator, _, goal = fixture()
        entries = {f"source:{index}": (None, (index * 8, 0, index * 8 + 8, 160), ())
                   for index in range(64)}
        scenarios = _discovery_scenarios(entries, simulator.map.rooms["source"], goal)
        self.assertIs(iter(scenarios), scenarios, "Pair combinations must be produced incrementally")
        prefix = list(islice(scenarios, 67))
        self.assertIn(frozenset(entries), prefix)
        self.assertEqual({next(iter(row)) for row in prefix if len(row) == 1}, set(entries))
        self.assertEqual(len(prefix[-1]), 2)


if __name__ == "__main__":
    unittest.main()
