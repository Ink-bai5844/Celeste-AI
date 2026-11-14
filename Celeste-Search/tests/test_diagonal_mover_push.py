"""Solid.MoveTo resolves H then V, with fresh riders on each axis."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DT, SimState, Simulator, _f32, _zip_ease


def fixture(*, mirror=False, ox=0, oy=0, obstruction=False):
    transform = lambda x: ox + (256 - x if mirror else x)
    x = transform(80 if mirror else 64)
    target = transform(96 if mirror else 80)
    mover = Entity('zipMover', 1, x, oy+96, 16, 24, {}, ((target, oy+48),))
    p = _zip_ease(16, 30)
    ex, ey = _f32(x+(target-x)*p), _f32(oy+96-48*p)
    entities = [mover]
    if obstruction:
        entities.append(Entity('solid', 2, ox+93, oy+64, 16, 64))
    room = Room('a', ox, oy, 256, 192, ('0'*32,)*24, tuple(entities), ())
    sim = Simulator(MapData('diagonal-solid', {'a': room}, Path('fixture.bin')))
    start = SimState('a', transform(89), oy+90,
                     mechanisms=(('a:1', 2, 16, ex, ey, 0.),))
    return sim, start


class DiagonalMoverPushTests(unittest.TestCase):
    def test_horizontal_push_clears_vertical_overlap_and_only_copies_horizontal_lift(self):
        for mirror, ox, oy in ((False, 0, 0), (True, -512, -192)):
            with self.subTest(mirror=mirror):
                sim, start = fixture(mirror=mirror, ox=ox, oy=oy)
                result = sim._advance_mechanisms(start)
                direction = -1 if mirror else 1
                self.assertEqual((result.x, result.y), (start.x+direction, start.y))
                self.assertEqual(result.lift_vy, 0.)
                self.assertGreater(result.lift_vx*direction, 0.)
                self.assertFalse(result.dead)
                self.assertEqual(start.mechanisms[0][2], 16)

    def test_top_and_side_riders_are_recomputed_after_horizontal_carry(self):
        sim, start = fixture()
        for state in (replace(start, x=77, y=80, vy=-10),
                      replace(start, y=92, state=1, facing=-1)):
            with self.subTest(state=state.state):
                result = sim._advance_mechanisms(state)
                self.assertEqual((result.x, result.y), (state.x+1, state.y-2))
                self.assertLess(result.lift_vy, 0.)
                self.assertFalse(result.dead)

    def test_vertical_only_overlap_still_pushes_and_copies_full_lift(self):
        sim, start = fixture()
        state = replace(start, x=77, y=79)
        result = sim._advance_mechanisms(state)
        self.assertEqual((result.x, result.y), (77, 78))
        self.assertGreater(result.lift_vx, 0.)
        self.assertLess(result.lift_vy, 0.)

    def test_horizontal_squish_is_not_hidden_by_later_vertical_separation(self):
        sim, start = fixture(obstruction=True)
        result = sim._advance_mechanisms(start)
        self.assertTrue(result.dead)
        self.assertEqual(result.reason, 'squish')
        self.assertEqual((result.x, result.y), (90, 90))

    def test_vertical_falling_rider_keeps_speed_and_landing_phase(self):
        block = Entity('fallingBlock', 1, 64, 64, 32, 24, {'climbFall': True})
        room = Room('a', 0, 0, 192, 192,
                    tuple('0'*24 if y<20 else '1'*24 for y in range(24)), (block,), ())
        sim = Simulator(MapData('vertical-falling', {'a': room}, Path('fixture.bin')))
        state = SimState('a', 80, 134, on_ground=True,
                         mechanisms=(('a:1', 2, 0, 64., 134., 160.),))
        result = sim._advance_mechanisms(state)
        self.assertEqual(result.y, 136)
        self.assertEqual(result.mechanisms[0][1:3], (5, 12))
        self.assertEqual(result.lift_vx, 0.)
        self.assertEqual(result.lift_vy, _f32(_f32(160*DT)/DT))
        self.assertFalse(result.dead)


if __name__ == '__main__':
    unittest.main()
