from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.recovery import _future_intercept, plan_recovery


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        rows = ("0" * 32,) * 18 + ("1" * 32,) * 2
        spring = Entity("spring", 7, 120, 144, 0, 0)
        room = Room("fixture", 0, 0, 256, 160, rows, (spring,), ((120.0, 144.0),), {})
        data = MapData("fixture", {room.name: room}, Path("fixture.bin"))
        sim = Simulator(data)
        launched = sim.step(sim.initial_state(), 0)
        self.assertFalse(launched.on_ground)
        self.assertEqual(launched.vy, -185)
        return data, sim, launched

    def test_airborne_spring_recovery_finishes_on_stable_ground(self):
        data, sim, start = self.fixture()
        with patch("celeste_search.recovery._context", return_value=(data, sim)):
            plan = plan_recovery("fixture.bin", start.to_dict(), seconds=1.5, expansions=1000)
        self.assertTrue(plan["can_execute"], plan)
        self.assertFalse(plan["requires_periodic_anchor"])
        self.assertTrue(plan["requires_future_anchor"])
        self.assertEqual(plan["observation_anchor_state"], start.to_dict())
        self.assertGreaterEqual(plan["execute_at_frame"] - start.frame, 180)
        self.assertLessEqual(plan["execute_at_frame"] - start.frame, 270)
        future = sim.roll(start, 0, plan["execute_at_frame"] - start.frame)
        self.assertEqual(plan["execution_anchor_state"], future.to_dict())
        self.assertEqual(plan["anchor_state"], future.to_dict())
        actual = sim.trace(future, plan["actions"])
        self.assertFalse(any(state.dead for state in actual))
        end = actual[-1]
        self.assertTrue(end.on_ground)
        self.assertGreaterEqual(abs(end.x - 120), 20)
        for state in sim.trace(end, [0] * 60):
            self.assertEqual((state.x, state.y, state.vx, state.vy), (end.x, end.y, 0, 0))
            self.assertTrue(state.on_ground)

    def test_descending_and_rising_snapshots_get_exact_future_anchors(self):
        data, sim, rising = self.fixture()
        descending = rising
        for _ in range(120):
            descending = sim.step(descending, 0)
            if descending.vy > 100:
                break
        self.assertGreater(descending.vy, 100)
        for observed in (rising, descending):
            with self.subTest(vy=observed.vy):
                future, count, reason = _future_intercept(sim, observed)
                self.assertIsNotNone(future, reason)
                self.assertEqual(future, sim.roll(observed, 0, count))
                self.assertEqual(future.frame, observed.frame + count)
                with patch("celeste_search.recovery._context", return_value=(data, sim)):
                    plan = plan_recovery("fixture.bin", observed.to_dict(), seconds=1, expansions=1000)
                self.assertTrue(plan["can_execute"], plan)
                exact = SimState.from_dict(plan["execution_anchor_state"])
                self.assertEqual((exact.rem_x, exact.rem_y), (future.rem_x, future.rem_y))

    def test_dead_or_dash_anchor_is_not_manufactured_into_a_recovery(self):
        data, sim, start = self.fixture()
        with patch("celeste_search.recovery._context", return_value=(data, sim)):
            for invalid in (replace(start, dead=True), replace(start, state=2)):
                plan = plan_recovery("fixture.bin", invalid.to_dict(), seconds=.2, expansions=10)
                self.assertFalse(plan["can_execute"])
                self.assertEqual(plan["actions"], [])


if __name__ == "__main__":
    unittest.main()
