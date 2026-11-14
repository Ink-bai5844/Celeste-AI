"""Only clear final boundary events may enter a locally unsupported interior."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist_targets import room_exit_targets
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import RIGHT, SimState, Simulator
from celeste_search.terminal_entry import (terminal_entry_certificate, terminal_entry_crossing_allowed,
                                           terminal_entry_handoff_safe)


def fixture(entities=None, *, source_entities=(), destination_attrs=None):
    if entities is None:
        entities = (Entity("cassetteBlock", 1, 120, 8, 16, 16),
                    Entity("cassette", 2, 144, 48, 0, 0))
    source = Room("a", 0, 0, 64, 64, ("0" * 8,) * 7 + ("1" * 8,),
                  source_entities, ((16.0, 56.0),))
    destination = Room("b", 64, 0, 96, 64, ("0" * 12,) * 7 + ("1" * 12,),
                       entities, ((72.0, 56.0),), destination_attrs or {})
    data = MapData("terminal", {"a": source, "b": destination}, Path("terminal.bin"))
    sim = Simulator(data)
    target = next(item for item in room_exit_targets(data, "a")
                  if item["edge"] == "right" and item["y"] == 24)
    return sim, target


class TerminalEntryTests(unittest.TestCase):
    def test_release_must_stop_on_permanent_floor_instead_of_sliding_off(self):
        sim, target = fixture((Entity("cassetteBlock", 1, 144, 8, 16, 16),))
        rooms = dict(sim.map.rooms)
        rooms["b"] = replace(rooms["b"], solids=("0" * 12,) * 7 + ("111000000000",))
        sim = Simulator(replace(sim.map, rooms=rooms))
        certificate = terminal_entry_certificate(sim, target, is_terminal=True)
        self.assertIsNotNone(certificate)
        safe = SimState("b", 69, 34, reason="transition")
        unsafe = replace(safe, state=2, vx=170, vy=170, dash_frames=12,
                         dash_dx=1, dash_dy=1, dashes=0)
        self.assertTrue(terminal_entry_handoff_safe(sim, certificate, safe))
        self.assertFalse(terminal_entry_handoff_safe(sim, certificate, unsafe))

    def test_clear_cassette_room_allows_only_actual_last_crossing(self):
        sim, target = fixture()
        certificate = terminal_entry_certificate(sim, target, is_terminal=True)
        self.assertIsNotNone(certificate)
        self.assertTrue(sim.capability_report("b")["unsupported"])
        before = SimState("a", 60, 34, vx=240)
        after = sim.step(before, RIGHT)
        self.assertEqual((after.room, after.reason), ("b", "transition"))
        self.assertTrue(terminal_entry_crossing_allowed(sim, certificate, before, after))
        self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, after, after))
        self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, before,
                                                       replace(after, reason="")))
        self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, before,
                                                       replace(after, y=46)))
        self.assertFalse(terminal_entry_crossing_allowed(sim, certificate, before,
                                                       replace(after, dead=True)))
        # This permission never changes the room's overall capabilities.
        self.assertFalse(sim.capability_report("b")["planning_supported"])

    def test_nonterminal_interior_forged_lane_and_unsupported_source_are_rejected(self):
        sim, target = fixture()
        self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=False))
        self.assertIsNone(terminal_entry_certificate(sim, dict(room="b", x=96, y=24), is_terminal=True))
        self.assertIsNone(terminal_entry_certificate(sim, dict(target, arrival_bounds=[64, 0, 160, 64]),
                                                    is_terminal=True))
        source_bad, other = fixture(source_entities=(Entity("dreamBlock", 9, 8, 8, 16, 16),))
        self.assertIsNone(terminal_entry_certificate(source_bad, other, is_terminal=True))

    def test_nearby_cassette_block_and_collectible_envelopes_are_rejected(self):
        for entity in (Entity("cassetteBlock", 1, 80, 24, 16, 16),
                       Entity("cassette", 2, 96, 32, 0, 0),
                       Entity("cassetteBlock", 3, 120, 8, 0, 16)):
            with self.subTest(kind=entity.kind, identity=entity.id):
                sim, target = fixture((entity,))
                self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=True))

    def test_other_unknown_entities_and_global_room_effects_stay_rejected(self):
        cases = [fixture((Entity("dreamBlock", 1, 120, 8, 16, 16),)),
                 fixture(destination_attrs={"windPattern": "Left"}),
                 fixture(destination_attrs={"underwater": True}),
                 fixture((Entity("cassetteBlock", 1, 120, 8, 16, 16),
                          Entity("eventTrigger", 2, 120, 24, 16, 16)))]
        for sim, target in cases:
            self.assertIsNone(terminal_entry_certificate(sim, target, is_terminal=True))

    def test_certificate_is_map_bound_and_known_arrival_hazards_still_apply(self):
        sim, target = fixture()
        certificate = terminal_entry_certificate(sim, target, is_terminal=True)
        before = SimState("a", 60, 34, vx=240)
        after = sim.step(before, RIGHT)
        other, _ = fixture()
        self.assertFalse(terminal_entry_crossing_allowed(other, certificate, before, after))
        hazard_sim, hazard_target = fixture((Entity("cassetteBlock", 1, 120, 8, 16, 16),
                                            Entity("spikesUp", 3, 64, 32, 16, 3)))
        hazard_certificate = terminal_entry_certificate(hazard_sim, hazard_target, is_terminal=True)
        self.assertIsNotNone(hazard_certificate)
        self.assertFalse(terminal_entry_crossing_allowed(hazard_sim, hazard_certificate, before, after))


if __name__ == "__main__":
    unittest.main()
