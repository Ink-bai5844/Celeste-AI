"""Native Solid pushes grant the same lift grace as carrying a rider."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator, _f32, _zip_ease, DT


def fixture(dx, dy):
    mover = Entity("zipMover", 1, 96, 96, 16, 16, {}, ((96 + dx, 96 + dy),))
    room = Room("a", 0, 0, 256, 256, ("0" * 32,) * 32,
                (mover,), ((24.0, 200.0),))
    sim = Simulator(MapData("push", {"a": room}, Path("push.bin")))
    old_x = _f32(96 + dx * _zip_ease(9, 30))
    old_y = _f32(96 + dy * _zip_ease(9, 30))
    return sim, old_x, old_y


class SolidPushLiftTests(unittest.TestCase):
    def test_horizontal_non_rider_push_inherits_lift_both_directions(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                sim, ex, ey = fixture(direction * 64, 0)
                x = round(ex) + (20 if direction > 0 else -4)
                start = SimState("a", x, round(ey) + 12, state=0,
                                 mechanisms=(("a:1", 2, 9, ex, ey, 0.0),))
                pushed = sim._advance_mechanisms(start)
                nx, ny = pushed.mechanisms[0][3:5]
                self.assertEqual(pushed.x, round(nx) + (20 if direction > 0 else -4))
                self.assertGreater((pushed.x - start.x) * direction, 0)
                self.assertEqual((pushed.y, pushed.state, pushed.lift_timer), (start.y, 0, 10))
                self.assertAlmostEqual(pushed.lift_vx, (nx - ex) / DT, places=4)
                self.assertEqual(pushed.lift_vy, 0)
                # Lift persists after separation, without applying it as speed.
                detached = sim._advance_mechanisms(replace(pushed, x=24, y=200))
                self.assertEqual((detached.lift_vx, detached.lift_timer, detached.vx),
                                 (pushed.lift_vx, 9, 0))

    def test_vertical_non_rider_push_inherits_lift_both_directions(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                sim, ex, ey = fixture(0, direction * 64)
                # Upward actor has negative vy, so native rider tests do not
                # mistake an upward solid catching its feet for standing.
                y = round(ey) + (27 if direction > 0 else 0)
                start = SimState("a", round(ex) + 8, y, state=0, vy=-1,
                                 mechanisms=(("a:1", 2, 9, ex, ey, 0.0),))
                pushed = sim._advance_mechanisms(start)
                nx, ny = pushed.mechanisms[0][3:5]
                self.assertEqual(pushed.y, round(ny) + (27 if direction > 0 else 0))
                self.assertGreater((pushed.y - start.y) * direction, 0)
                self.assertEqual((pushed.x, pushed.state, pushed.lift_timer), (start.x, 0, 10))
                self.assertAlmostEqual(pushed.lift_vy, (ny - ey) / DT, places=4)
                self.assertEqual(pushed.lift_vx, 0)

    def test_remote_solid_does_not_refresh_lift(self):
        sim, ex, ey = fixture(64, 0)
        start = SimState("a", 24, 200, lift_vx=30, lift_timer=4,
                         mechanisms=(("a:1", 2, 9, ex, ey, 0.0),))
        moved = sim._advance_mechanisms(start)
        self.assertEqual((moved.x, moved.y, moved.lift_vx, moved.lift_timer), (24, 200, 30, 3))


if __name__ == "__main__":
    unittest.main()
