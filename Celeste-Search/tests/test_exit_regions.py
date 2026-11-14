"""Connected exits retain physical lane and final-entry safety boundaries."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist_routes import normalize_route, start_route_progress, advance_route_progress
from celeste_search.assist_targets import expand_exit_target, room_exit_targets, transition_target_reached
from celeste_search.assist_validation import AssistValidationError, validate_tile_target
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import RIGHT, SimState, Simulator
from celeste_search.terminal_entry import (
    terminal_entry_certificate, terminal_entry_crossing_allowed, terminal_entry_handoff_safe,
)


def room(name, x, y, *, width=64, height=64, entities=(), floor=False):
    rows = ("0" * (width // 8),) * (height // 8)
    if floor:
        rows = rows[:-1] + ("1" * (width // 8),)
    return Room(name, x, y, width, height, rows, entities, ((x + 16., y + height - 8.),))


def map_of(*rooms):
    return MapData("region-fixture", {item.name: item for item in rooms}, Path("region-fixture.bin"))


def select(data, edge, *, x=None, y=None, destination=None):
    return next(item for item in room_exit_targets(data, "source")
                if item["edge"] == edge and (x is None or item["x"] == x)
                and (y is None or item["y"] == y)
                and (destination is None or item["transition_to"] == destination))


class ExitRegionTests(unittest.TestCase):
    def test_region_accepts_other_native_lane_on_all_edges_but_legacy_target_stays_exact(self):
        data = map_of(room("source", 0, 0), room("left", -64, 0), room("right", 64, 0),
                      room("top", 0, -64), room("bottom", 0, 64))
        simulator = Simulator(data)
        cases = (("left", dict(y=16), (3, 46)), ("right", dict(y=16), (61, 46)),
                 ("top", dict(x=16), (42, 5)), ("bottom", dict(x=16), (42, 65)))
        for edge, coordinate, position in cases:
            with self.subTest(edge=edge):
                exact = select(data, edge, **coordinate)
                region = expand_exit_target(data, exact)
                before = SimState("source", *position)
                after = simulator._bounds(before)
                self.assertEqual(after.room, edge)
                self.assertFalse(transition_target_reached(after, exact))
                self.assertTrue(transition_target_reached(after, region))
                self.assertEqual(validate_tile_target(data, region), region)
                self.assertEqual(validate_tile_target(data, exact), exact)
                route = normalize_route(data, {"waypoints": [region]})
                progress = start_route_progress(route, before)
                self.assertEqual(advance_route_progress(route, progress, before, after, 0).index, 1)
                self.assertEqual(start_route_progress(route, after).index, 0)

    def test_adjacent_partial_tiles_do_not_connect_through_thin_permanent_barrier(self):
        barrier = Entity("solid", 1, 56, 32, 16, 1)
        data = map_of(room("source", 0, 0, entities=(barrier,)), room("destination", 64, 0))
        upper = expand_exit_target(data, select(data, "right", y=24))
        lower = expand_exit_target(data, select(data, "right", y=32))
        self.assertEqual(max(member["y"] for member in upper["exit_region"]), 24)
        self.assertEqual(min(member["y"] for member in lower["exit_region"]), 32)
        self.assertLess(max(member["arrival_bounds"][3] for member in upper["exit_region"]),
                        min(member["arrival_bounds"][1] for member in lower["exit_region"]))
        lower_arrival = Simulator(data)._bounds(SimState("source", 61, 44))
        self.assertTrue(transition_target_reached(lower_arrival, lower))
        self.assertFalse(transition_target_reached(lower_arrival, upper))
        # The actual blocked strip is never filled by an enclosing rectangle.
        blocked = SimState("destination", 69, 37.5)
        self.assertFalse(transition_target_reached(blocked, upper))
        self.assertFalse(transition_target_reached(blocked, lower))

    def test_region_does_not_merge_neighboring_destinations(self):
        data = map_of(room("source", 0, 0), room("upper", 64, 0, height=32),
                      room("lower", 64, 32, height=32))
        region = expand_exit_target(data, select(data, "right", y=16, destination="upper"))
        self.assertTrue(all(item["y"] < 32 for item in region["exit_region"]))
        after = Simulator(data)._bounds(SimState("source", 61, 46))
        self.assertEqual(after.room, "lower")
        self.assertFalse(transition_target_reached(after, region))

    def test_validation_rederives_region_and_never_trusts_saved_expanded_bounds(self):
        data = map_of(room("source", -128, -64), room("destination", -64, -64))
        exact = select(data, "right", y=-48)
        region = expand_exit_target(data, exact)
        forged = deepcopy(region)
        forged["exit_region"][0]["arrival_bounds"] = [-999, -999, 999, 999]
        forged["exit_region"].append(dict(x=999, y=999, arrival_bounds=[0, 0, 999, 999]))
        self.assertEqual(validate_tile_target(data, forged), region)
        for invalid in (None, False, {}, []):
            with self.subTest(invalid=invalid), self.assertRaises(AssistValidationError):
                validate_tile_target(data, dict(exact, exit_region=invalid))
        with self.assertRaises(AssistValidationError):
            validate_tile_target(data, dict(room="source", x=-112, y=-48, exit_region=region["exit_region"]))
        # Caller edits cannot poison geometry cached for later selections.
        region["exit_region"][0]["arrival_bounds"][0] = -999
        self.assertNotEqual(expand_exit_target(data, exact)["exit_region"][0]["arrival_bounds"][0], -999)


def terminal_fixture(viewer_y=24):
    data = map_of(room("source", 0, 0, height=96, floor=True),
                  room("destination", 64, 0, width=128, height=96, floor=True,
                       entities=(Entity("towerviewer", 1, 96, viewer_y, 0, 0),)))
    simulator = Simulator(data)
    region = expand_exit_target(data, select(data, "right", y=56))
    return simulator, region


class RegionalTerminalEntryTests(unittest.TestCase):
    def test_safe_lane_does_not_authorize_another_lane_near_unknown_mechanics(self):
        simulator, region = terminal_fixture()
        certificate = terminal_entry_certificate(simulator, region, is_terminal=True)
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate.lanes)
        for y, allowed in ((66, True), (18, False)):
            with self.subTest(y=y):
                before = SimState("source", 60, y, vx=240)
                after = simulator.step(before, RIGHT)
                self.assertEqual(after.room, "destination")
                self.assertTrue(transition_target_reached(after, region))
                self.assertEqual(terminal_entry_crossing_allowed(simulator, certificate, before, after), allowed)
                self.assertEqual(terminal_entry_handoff_safe(simulator, certificate, after), allowed)
                self.assertEqual(transition_target_reached(after, certificate.target()), allowed)
        self.assertFalse(simulator.capability_report("destination")["planning_supported"])
        self.assertIsNone(terminal_entry_certificate(simulator, region, is_terminal=False))

    def test_safe_initial_lane_still_requires_passive_handoff_away_from_unknown(self):
        simulator, region = terminal_fixture(viewer_y=56)
        certificate = terminal_entry_certificate(simulator, region, is_terminal=True)
        before = SimState("source", 60, 18, vx=240)
        after = simulator.step(before, RIGHT)
        self.assertTrue(transition_target_reached(after, certificate.target()))
        self.assertFalse(terminal_entry_handoff_safe(simulator, certificate, after))
        self.assertFalse(terminal_entry_crossing_allowed(simulator, certificate, before, after))

    def test_regional_certificate_rejects_forged_members_and_other_map(self):
        simulator, region = terminal_fixture()
        forged = deepcopy(region)
        forged["exit_region"][0]["arrival_bounds"][3] += 1
        self.assertIsNone(terminal_entry_certificate(simulator, forged, is_terminal=True))
        self.assertIsNone(terminal_entry_certificate(simulator, dict(region, exit_region=[]), is_terminal=True))
        certificate = terminal_entry_certificate(simulator, region, is_terminal=True)
        other, _ = terminal_fixture()
        before = SimState("source", 60, 66, vx=240)
        after = simulator.step(before, RIGHT)
        self.assertFalse(terminal_entry_crossing_allowed(other, certificate, before, after))
        self.assertFalse(terminal_entry_crossing_allowed(simulator, certificate, after, after))
        self.assertFalse(terminal_entry_crossing_allowed(simulator, certificate, before, replace(after, dead=True)))


if __name__ == "__main__":
    unittest.main()
