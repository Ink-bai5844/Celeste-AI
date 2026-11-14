"""Native jump-through boost follows actual climbing, not the held Up input."""
from dataclasses import replace
from pathlib import Path
import struct
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, LEFT, UP, SimState, Simulator


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def fixture(*, ceiling=False):
    entities = [Entity("solid", 1, 0, 48, 8, 48),
                Entity("jumpThru", 2, 8, 48, 16, 0)]
    if ceiling:
        entities.append(Entity("solid", 3, 8, 40, 16, 8))
    room = Room("a", 0, 0, 64, 160, (), tuple(entities), ((12.0, 63.0),), {})
    return Simulator(MapData("climb-boost", {room.name: room}, Path("fixture.bin")))


class ClimbJumpThruBoostTests(unittest.TestCase):
    def test_new_grab_does_not_boost_from_held_up(self):
        sim = fixture()
        start = SimState("a", 12, 63, facing=-1, rem_y=f32(.4))
        grabbed = sim.step(start, LEFT | UP | GRAB)
        self.assertEqual((grabbed.state, grabbed.climb_wait, grabbed.y), (1, 6, 63))
        self.assertEqual(grabbed.rem_y, start.rem_y)

    def test_wait_then_actual_upward_climb_matches_native_remainders(self):
        sim = fixture()
        # Mac request1 native observations at player frames 53653..53664,
        # translated from wall top -2128 to fixture top 48. No game file needed.
        state = SimState("a", 12, 63, facing=-1, state=1, climb_wait=6,
                         vy=f32(4.234314e-5), rem_y=f32(.499966621))
        observed = [
            (63, 0, .499966621, 5),
            (63, 0, .499966621, 4),
            (63, 0, .499966621, 3),
            (63, 0, .499966621, 2),
            (63, 0, .499966621, 1),
            (63, -15.00003, -.4167023, 0),
            (61, -30.00006, .416627824, 0),
            (60, -45, -4.16040421e-5, 0),
            (59, -45, -.416711032, 0),
            (57, -45, .16661948, 0),
            (56, -45, -.250049949, 0),
        ]
        for index, (y, vy, remainder, wait) in enumerate(observed, 1):
            state = sim.step(state, LEFT | UP | GRAB)
            with self.subTest(update=index):
                self.assertEqual((state.x, state.y, state.state, state.climb_wait), (12, y, 1, wait))
                self.assertAlmostEqual(state.vy, vy, places=5)
                self.assertAlmostEqual(state.rem_y, remainder, places=6)

    def test_ceiling_blocked_up_climb_does_not_boost(self):
        sim = fixture(ceiling=True)
        start = SimState("a", 12, 59, facing=-1, state=1, climb_wait=0,
                         vy=-30, rem_y=f32(.3))
        blocked = sim.step(start, LEFT | UP | GRAB)
        self.assertEqual((blocked.y, blocked.vy, blocked.state), (59, 0, 1))
        self.assertEqual(blocked.rem_y, start.rem_y)
        self.assertAlmostEqual(blocked.stamina, 110-10*f32(.0166667), places=5)


if __name__ == "__main__":
    unittest.main()
