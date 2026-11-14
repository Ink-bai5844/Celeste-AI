"""The alternative mechanism arm gets capacity without bypassing raw replay."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist import TargetPlanner
from celeste_search.assist_planning import plan_to_route
from celeste_search.assist_routes import (
    RouteProgress, advance_route_progress, route_complete, start_route_progress,
)
from celeste_search.assist_strategies import route_strategy_specs
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, RIGHT, SimState, Simulator
from celeste_search.routes import snapshot


ARMS = {"ordered_mechanism", "ordered_mechanism_capability"}


class RecordingPool:
    def __init__(self):
        self.calls = []

    def submit(self, function, config):
        future = Future()
        self.calls.append((deepcopy(config), future))
        return future

    def shutdown(self, **kwargs):
        pass


def fixture(with_zip=True):
    entities = ((Entity("zipMover", 1, 128, 16, 16, 16,
                        {}, ((144., 16.),)),) if with_zip else ())
    room = Room("a", 0, 0, 176, 80,
                ("0"*22,)*8+("1"*22,)*2, entities, ((16., 64.),))
    data = MapData("mechanism-arm-contract", {"a": room}, Path("arm-contract.bin"))
    sim = Simulator(data)
    return data, sim, sim.initial_state("a")


def public_route(two=False):
    points = [dict(room="a", x=48, y=56, tile_size=8)]
    if two:
        points.append(dict(room="a", x=96, y=56, tile_size=8))
    return dict(waypoints=points,
                segments=[dict(no_dash=False, no_ground=False) for _ in points])


class MechanismCapabilityPortfolioTests(unittest.TestCase):
    def schedule(self, data, start, route):
        pool = RecordingPool()
        planner = TargetPlanner(workers=5, seconds=8, expansions=10000,
                                pool=pool, clock=lambda: 100.)
        self.addCleanup(planner.close)
        planner.set_map(data)
        planner.set_deadline(190.)
        self.assertTrue(planner.request(str(data.source), start, route, 0, []))
        return planner, [config for config, _ in pool.calls]

    def test_both_zip_arms_reach_rolling_queue_with_equal_full_budgets(self):
        data, _, start = fixture()
        planner, jobs = self.schedule(data, start, public_route())
        self.assertEqual(len(jobs), 5)
        # Corridor has its own first slot. Other families still get diverse
        # capacity, and the second ordered arm must remain queued, not filtered.
        self.assertEqual(jobs[0]["strategy"],"ordered_corridor")
        families = {row["name"]: row.get("portfolio_family",row["family"]) for row in planner.specs}
        self.assertEqual(len({families[job["strategy"]] for job in jobs}), 5)
        self.assertTrue(ARMS <= {row["name"] for row in planner.specs})
        self.assertTrue({"direct", "support_graph", "zip_graph"} <= set(families.values()))
        for _ in range(2):
            for _, future in tuple(planner.pool.calls):
                if not future.done():
                    future.set_result(dict(can_execute=False, reached=False, reason="fixture_failed"))
            planner.poll()
        selected = {config["strategy"]: config for config, _ in planner.pool.calls}
        self.assertTrue(ARMS <= selected.keys(), list(selected))
        # One new corridor slot may move the second arm from tenth to eleventh;
        # it must not fall behind the rest of the large ordered recipe inventory.
        order=list(selected)
        self.assertLessEqual(max(order.index(name) for name in ARMS),2*planner.workers)
        for name in ARMS:
            with self.subTest(strategy=name):
                self.assertEqual((selected[name]["seconds"], selected[name]["expansions"]),
                                 (90, 25000))
        self.assertEqual(planner.deadline, 190.)
        specs = {row["name"]: row for row in route_strategy_specs(public_route(), start, data)}
        self.assertEqual({(specs[name]["budget_scale"], specs[name]["time_budget_scale"],
                           specs[name]["max_seconds"]) for name in ARMS}, {(2.5, 11.25, 90)})

    def test_no_zip_filters_both_arms_from_inventory_and_actual_queue(self):
        data, _, start = fixture(with_zip=False)
        route = public_route()
        specs = route_strategy_specs(route, start, data)
        planner, jobs = self.schedule(data, start, route)
        self.assertFalse(ARMS & {row["name"] for row in specs})
        self.assertFalse(ARMS & {row["name"] for row in planner.specs})
        self.assertFalse(ARMS & {row["strategy"] for row in jobs})

    def test_breakable_block_without_mover_enables_active_mechanism_search(self):
        data, _, start = fixture(with_zip=False)
        block = Entity("dashBlock", 7, 80, 32, 16, 32, {"canDash": True})
        room = replace(data.rooms["a"], entities=(block,))
        data = replace(data, rooms={"a": room})
        planner, jobs = self.schedule(data, start, public_route())
        self.assertTrue(ARMS <= {row["name"] for row in planner.specs})
        self.assertIn("ordered_mechanism", {row["strategy"] for row in jobs})
        self.assertEqual(jobs[0]["strategy"], "ordered_corridor")
        self.assertTrue(all(job["start_state"] == start.to_dict() for job in jobs))

    def test_unbreakable_or_removed_block_does_not_enable_extra_arms(self):
        for can_dash, broken in ((False, ()), ("false", ()), (True, ("a:7",))):
            with self.subTest(can_dash=can_dash, broken=broken):
                data, _, start = fixture(with_zip=False)
                block = Entity("dashBlock", 7, 80, 32, 16, 32, {"canDash": can_dash})
                data = replace(data, rooms={"a": replace(data.rooms["a"], entities=(block,))})
                start = replace(start, broken=broken)
                specs = route_strategy_specs(public_route(), start.to_dict(), data)
                self.assertFalse(ARMS & {row["name"] for row in specs})

    def test_dash_only_mechanism_arms_respect_all_no_dash_segments(self):
        data, _, start = fixture(with_zip=False)
        block = Entity("dashBlock", 7, 80, 32, 16, 32, {"canDash": True})
        data = replace(data, rooms={"a": replace(data.rooms["a"], entities=(block,))})
        route = public_route(two=True)
        for segment in route["segments"]:
            segment["no_dash"] = True
        self.assertFalse(ARMS & {row["name"] for row in route_strategy_specs(route, start, data)})
        route["segments"][1]["no_dash"] = False
        self.assertTrue(ARMS <= {row["name"] for row in route_strategy_specs(route, start, data)})

    def test_restricted_routes_keep_both_arms_in_ordered_family(self):
        data, _, start = fixture()
        for route in (public_route(two=True), public_route()):
            route["segments"][0]["no_dash"] = True
            with self.subTest(points=len(route["waypoints"])):
                specs = route_strategy_specs(route, start, data)
                self.assertTrue(all(row["family"] == "ordered" for row in specs))
                self.assertTrue(ARMS <= {row["name"] for row in specs})
                _, jobs = self.schedule(data, start, route)
                self.assertTrue(ARMS <= {row["strategy"] for row in jobs})


class MechanismCapabilityReplayTests(unittest.TestCase):
    def setUp(self):
        self.data, self.sim, self.start = fixture()
        self.route = public_route(two=True)
        self.context = patch("celeste_search.assist_planning._context",
                             return_value=(self.data, self.sim))
        self.context.start()
        self.addCleanup(self.context.stop)

    def walk(self):
        state, phase = self.start, start_route_progress(self.route, self.start)
        actions, states, indices = [], [snapshot(state)], []
        for _ in range(100):
            before, old_index = state, phase.index
            state = self.sim.step(state, RIGHT)
            actions.append(RIGHT)
            states.append(snapshot(state))
            phase = advance_route_progress(self.route, phase, before, state, RIGHT)
            if phase.index != old_index:
                indices.append(len(actions))
            if route_complete(self.route, phase):
                return actions, state, phase, states, indices
        self.fail("Walking fixture must physically complete both original targets")

    def invoke(self, answer, *, strategy="ordered_mechanism_capability", route=None):
        route = self.route if route is None else route
        saved = deepcopy(route)
        with patch("celeste_search.mechanism_planning.search_mechanism_routes",
                   return_value=answer) as solver, \
                patch.object(self.sim, "initial_state", side_effect=AssertionError("state reset")):
            result = plan_to_route(str(self.data.source), self.start.to_dict(), route,
                                   strategy=strategy, seconds=1, max_attempts=1)
        solver.assert_called_once()
        self.assertEqual(solver.call_args.args[1:3], (self.start, saved))
        self.assertEqual(solver.call_args.kwargs["use_detours"], strategy == "ordered_mechanism")
        self.assertEqual(route, saved)
        return result

    def test_new_dispatch_preserves_original_route_and_complete_raw_replay(self):
        actions, state, phase, expected, indices = self.walk()
        for strategy in sorted(ARMS):
            with self.subTest(strategy=strategy):
                result = self.invoke((actions, state, True, 1, len(actions), phase, {}),
                                     strategy=strategy)
                self.assertTrue(result["can_execute"], result)
                self.assertEqual(result["route"], self.route)
                self.assertEqual(result["actions"], actions)
                self.assertEqual(result["expected"], expected)
                self.assertEqual(result["waypoint_indices"], indices)
                self.assertEqual(SimState.from_dict(result["anchor_state"]), self.start)
                self.assertEqual(SimState.from_dict(result["final_state"]), state)

    def test_claimed_complete_prefix_cannot_authorize_partial_route(self):
        actions, state, phase, _, _ = self.walk()
        invented = replace(state, dashes=2, stamina=110.)
        result = self.invoke((actions[:3], invented, True, 1, len(actions), phase, {}))
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"], [])
        self.assertTrue(result["attempts"][0]["replay_rejected"])

    def test_capability_arm_cannot_relax_public_segment_constraints(self):
        actions, state, _, _, _ = self.walk()
        for flag in ("no_dash", "no_ground"):
            route = deepcopy(self.route)
            route["segments"][0][flag] = True
            unsafe = actions.copy()
            if flag == "no_dash":
                unsafe[0] |= DASH
            with self.subTest(constraint=flag):
                result = self.invoke((unsafe, state, True, 1, len(unsafe), RouteProgress(2), {}),
                                     route=route)
                self.assertFalse(result["can_execute"])
                self.assertEqual(result["actions"], [])
                self.assertTrue(result["attempts"][0]["replay_rejected"])


if __name__ == "__main__":
    unittest.main()
