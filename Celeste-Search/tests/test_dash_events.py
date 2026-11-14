"""Exact per-step dash events, including when returned state hides consumption."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, RIGHT, SimState, Simulator


def simulator(entities=()):
    room = Room("a", 0, 0, 160, 80, ("0"*20,)*6 + ("1"*20,)*4,
                tuple(entities), ((16.0, 48.0),), {})
    return Simulator(MapData("dash-events", {room.name: room}, Path("fixture.bin")))


class DashEventTests(unittest.TestCase):
    def test_buffered_dash_after_ground_refill_counts_zero_to_zero(self):
        sim = simulator()
        falling = SimState("a", 16, 47, vy=60, dashes=0)
        landed = sim.step(falling, DASH)
        self.assertEqual((landed.y, landed.dashes, landed.state), (48, 0, 0))
        self.assertFalse(sim.last_step_started_dash)
        self.assertGreater(landed.dash_buffer, 0)
        dashed = sim.step(landed, 0)
        self.assertEqual((dashed.dashes, dashed.state), (0, 2))
        self.assertTrue(sim.last_step_started_dash)

    def test_overlapping_refill_counts_one_to_one(self):
        sim = simulator([Entity("refill", 1, 16, 42, 0, 0)])
        start = sim.initial_state()
        dashed = sim.step(start, DASH)
        self.assertEqual((start.dashes, dashed.dashes, dashed.state), (1, 1, 2))
        self.assertTrue(sim.last_step_started_dash)
        self.assertIn(("a:1", 150), dashed.refill_timers)

    def test_same_tick_spring_can_replace_dash_state_without_erasing_event(self):
        sim = simulator([Entity("spring", 1, 16, 48, 0, 0)])
        start = sim.initial_state()
        bounced = sim.step(start, DASH)
        self.assertEqual((bounced.state, bounced.dashes, bounced.vy), (0, 1, -185))
        self.assertTrue(sim.last_step_started_dash)
        self.assertNotIn("last_step_started_dash", bounced.to_dict())

    def test_held_input_only_marks_the_start_and_next_call_resets(self):
        sim = simulator()
        self.assertFalse(sim.last_step_started_dash)
        start = replace(sim.initial_state(), y=24, on_ground=False, grace=0)
        dashed = sim.step(start, RIGHT | DASH)
        self.assertTrue(sim.last_step_started_dash)
        continued = sim.step(dashed, RIGHT | DASH)
        self.assertFalse(sim.last_step_started_dash)
        self.assertEqual(continued.state, 2)
        # Branching from an earlier state gets a new event, without mutating it.
        self.assertEqual(sim.step(start, RIGHT | DASH), dashed)
        self.assertTrue(sim.last_step_started_dash)
        sim.step(replace(start, dead=True), 0)
        self.assertFalse(sim.last_step_started_dash)

    def test_invalid_input_resets_previous_event_before_raising(self):
        sim = simulator()
        start = sim.initial_state()
        sim.step(start, DASH)
        self.assertTrue(sim.last_step_started_dash)
        with self.assertRaises(ValueError):
            sim.step(start, 128)
        self.assertFalse(sim.last_step_started_dash)


if __name__ == "__main__":
    unittest.main()
