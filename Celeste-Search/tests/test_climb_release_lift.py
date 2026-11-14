"""Moving-wall release retains native capped velocity without a jump."""
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, LEFT, RIGHT, SimState, Simulator


def simulator():
    wall = Entity("solid", 1, 96, 80, 16, 60)
    room = Room("a", 0, 0, 240, 184,
                ("0" * 30,) * 21 + ("1" * 30,) * 2,
                (wall,), ((24.0, 168.0),))
    return Simulator(MapData("release", {"a": room}, Path("release.bin")))


class ClimbReleaseLiftTests(unittest.TestCase):
    def test_let_go_adds_capped_lift_in_both_directions_without_jumping(self):
        for direction, x, facing, mask in ((1, 116, -1, RIGHT), (-1, 92, 1, LEFT)):
            with self.subTest(direction=direction):
                sim = simulator()
                start = SimState("a", x, 112, state=1, facing=facing,
                                 lift_timer=10, lift_vx=427 * direction, lift_vy=-200,
                                 dashes=0, var_speed=42)
                released = sim.step(start, mask)
                self.assertEqual((released.state, released.vx, released.vy), (0, 250 * direction, -130))
                self.assertEqual((released.var_jump, released.var_speed, released.dashes), (0, 42, 0))
                self.assertEqual(released.lift_timer, 9)
                self.assertEqual(start.vx, 0)

    def test_exhaustion_after_stamina_drain_applies_lift_once(self):
        sim = simulator()
        start = SimState("a", 116, 112, state=1, facing=-1, stamina=.01,
                         lift_timer=10, lift_vx=100, lift_vy=-200)
        exhausted = sim.step(start, GRAB)
        self.assertLessEqual(exhausted.stamina, 0)
        self.assertEqual((exhausted.state, exhausted.vx, exhausted.vy), (0, 100, -130))
        later = sim.step(exhausted, RIGHT)
        self.assertLess(later.vx, 100)
        self.assertGreater(later.vy, -130)


if __name__ == "__main__":
    unittest.main()
