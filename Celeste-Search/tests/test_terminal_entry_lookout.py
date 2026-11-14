"""A distant Lookout permits one safe terminal crossing, never room planning."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route
from celeste_search.assist_targets import room_exit_targets
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import LEFT, RIGHT, SimState, Simulator
from celeste_search.terminal_entry import (
    terminal_entry_certificate, terminal_entry_crossing_allowed,
    terminal_entry_handoff_safe,
)


TRANSFORMS = (("ordinary", 0, 0, False),
              ("translated", -824, -1616, False),
              ("mirrored", 456, -440, True))


def fixture(*, viewer=(160, 88), transform=TRANSFORMS[0], extra=()):
    name, ox, oy, mirror = transform
    source_name, destination_name = f"{name}-source", f"{name}-lookout"

    def px(x):
        return ox + (192-x if mirror else x)

    def entity(e):
        return replace(e, x=px(e.x+e.width) if mirror else px(e.x), y=oy+e.y,
                       nodes=tuple((px(x+e.width) if mirror else px(x), oy+y)
                                   for x, y in e.nodes))

    entities = (Entity("towerviewer", 1, *viewer, 0, 0),) + tuple(extra)
    source = Room(source_name, px(64) if mirror else px(0), oy, 64, 96,
                  ("0"*8,)*11+("1"*8,), (), ((px(16), oy+88),))
    destination = Room(destination_name, px(192) if mirror else px(64), oy, 128, 96,
                       ("0"*16,)*11+("1"*16,), tuple(map(entity, entities)),
                       ((px(136), oy+88),))
    data = MapData("lookout-terminal", {source.name: source, destination.name: destination},
                   Path("lookout-terminal.bin"))
    simulator = Simulator(data)
    target = next(row for row in room_exit_targets(data, source.name)
                  if row["edge"] == ("left" if mirror else "right") and row["y"] == oy+8)
    before = SimState(source.name, px(60), oy+18,
                      vx=-240 if mirror else 240, frame=371)
    action = LEFT if mirror else RIGHT
    return simulator, target, before, action, px, oy


class LookoutTerminalEntryTests(unittest.TestCase):
    def test_distant_viewer_allows_only_exact_final_crossing_under_mirror_and_translation(self):
        for transform in TRANSFORMS:
            with self.subTest(transform=transform[0]):
                sim, target, before, action, _, _ = fixture(transform=transform)
                certificate = terminal_entry_certificate(sim, target, is_terminal=True)
                self.assertIsNotNone(certificate)
                self.assertEqual(certificate.unsupported_kinds, ("towerviewer",))
                self.assertFalse(sim.capability_report(target["transition_to"])["planning_supported"])
                after = sim.step(before, action)
                self.assertEqual((after.room, after.reason), (target["transition_to"], "transition"))
                self.assertTrue(terminal_entry_crossing_allowed(sim, certificate, before, after))
                self.assertTrue(terminal_entry_handoff_safe(sim, certificate, after))
                self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=False))
                self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, after, after))

    def test_viewer_talk_envelope_near_entry_rejects_certificate(self):
        for transform in TRANSFORMS:
            with self.subTest(transform=transform[0]):
                # The zero-sized map entity itself is not in the entry lane;
                # its native Talk/body envelope extends left into that lane.
                sim, target, _, _, _, _ = fixture(viewer=(96, 24), transform=transform)
                self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=True))

    def test_clear_entry_but_neutral_fall_near_viewer_rejects_handoff(self):
        for transform in TRANSFORMS:
            with self.subTest(transform=transform[0]):
                sim, target, before, action, _, _ = fixture(viewer=(96, 72), transform=transform)
                certificate = terminal_entry_certificate(sim, target, is_terminal=True)
                self.assertIsNotNone(certificate, "Viewer is below the initial entry envelope")
                after = sim.step(before, action)
                self.assertEqual(after.room, target["transition_to"])
                self.assertFalse(terminal_entry_handoff_safe(sim, certificate, after))
                self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, before, after))

    def test_other_unknown_kinds_remain_rejected_even_when_far_away(self):
        for kind in ("dreamBlock", "eventTrigger", "unrecognizedFixtureEntity"):
            with self.subTest(kind=kind):
                sim, target, _, _, _, _ = fixture(extra=(Entity(kind, 9, 152, 64, 16, 16),))
                self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=True))

    def test_source_interior_and_intermediate_destination_targets_stay_unsupported(self):
        sim, target, before, _, px, oy = fixture()
        interior = dict(room=target["transition_to"], x=px(152), y=oy+80, tile_size=8)
        cases = ((before, {"waypoints": [target, interior]}),
                 (before, {"waypoints": [interior]}),
                 (SimState(target["transition_to"], px(136), oy+88, on_ground=True),
                  {"waypoints": [interior]}))
        for start, route in cases:
            with self.subTest(source=start.room, points=len(route["waypoints"])):
                with patch("celeste_search.assist_planning._context", return_value=(sim.map, sim)), \
                        patch.object(sim, "step", side_effect=AssertionError("unsupported interior search")):
                    result = plan_to_route(str(sim.map.source), start.to_dict(), route,
                                           strategy="ordered_semantic", seconds=1, max_attempts=1)
                self.assertEqual(result["reason"], "unsupported_room_mechanics")
                self.assertFalse(result["can_execute"])
                self.assertEqual(result["actions"], [])

    def test_planner_stops_on_boundary_without_simulating_destination_inputs(self):
        for transform in TRANSFORMS:
            with self.subTest(transform=transform[0]):
                sim, target, before, action, _, _ = fixture(transform=transform)
                native_step = sim.step

                def guarded_step(state, mask):
                    self.assertEqual(state.room, before.room, "No planned input after the boundary")
                    return native_step(state, mask)

                with patch("celeste_search.assist_planning._context", return_value=(sim.map, sim)), \
                        patch("celeste_search.planner.macros", return_value=[(action, 4)]), \
                        patch("celeste_search.assist_planning.macros", return_value=[(action, 4)]), \
                        patch.object(sim, "step", side_effect=guarded_step):
                    result = plan_to_route(str(sim.map.source), before.to_dict(),
                        {"waypoints": [target]}, strategy="ordered_semantic", seconds=1,
                        expansions=10, max_attempts=1)
                self.assertTrue(result["can_execute"], result)
                self.assertTrue(result["terminal_entry_only"])
                self.assertEqual(result["actions"], [action])
                self.assertEqual(result["waypoint_indices"], [1])
                self.assertEqual(result["final_state"]["room"], target["transition_to"])


if __name__ == "__main__":
    unittest.main()
