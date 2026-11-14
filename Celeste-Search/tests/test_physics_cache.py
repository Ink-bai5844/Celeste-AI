"""Collision-cache equivalence and branch isolation, without a game process."""
from dataclasses import replace
import math
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import RIGHT, SimState, Simulator


def fixture(entities=(), *, rows=("1000", "0010", "0100", "0001"), x=-24, y=-16):
    room = Room("test", x, y, 32, 32, rows, tuple(entities), ())
    return MapData("cache-fixture", {"test": room}, Path("fixture.bin"))


def original_tile_query(sim, rect):
    """Pre-cache implementation retained as a differential reference."""
    room = sim.map.rooms["test"]
    for ty in range(math.floor((rect[1]-room.y)/8), math.floor((rect[3]-room.y-1e-6)/8)+1):
        for tx in range(math.floor((rect[0]-room.x)/8), math.floor((rect[2]-room.x-1e-6)/8)+1):
            if (tx, ty) in sim._tiles["test"]:
                return True
    return any(rect[0]<r and rect[2]>l and rect[1]<b and rect[3]>t
               for l,t,r,b in sim._solids["test"])


class CollisionCacheTests(unittest.TestCase):
    def test_fractional_negative_and_outside_geometry_matches_original(self):
        sim = Simulator(fixture((Entity("solid", 1, -35.5, -6.25, 4.25, 9.75),)))
        offsets = (-1.000001, -1e-7, 0, 1e-7, .25, .999999)
        for x in range(-40, 25, 4):
            for y in range(-32, 33, 4):
                for offset in offsets:
                    px, py = x+offset, y-offset
                    for ducking in (False, True):
                        box = (px-4, py-(6 if ducking else 11), px+4, py)
                        self.assertEqual(sim.solid_at("test", px, py, ducking=ducking),
                                         original_tile_query(sim, box), (box, ducking))
                    box = (px-4, py-10, px+4, py+1)
                    self.assertEqual(sim._tiles_overlap("test", box), original_tile_query(sim, box))
        self.assertGreater(sim._tile_span_occupied.cache_info().hits, 1000)

    def test_mechanism_cache_separates_breaks_phases_and_subpixels(self):
        entities = (Entity("dashBlock", 1, 0, 0, 8, 8),
                    Entity("zipMover", 2, 8, 0, 8, 8, {}, ((16, 0),)),
                    Entity("crumbleBlock", 3, 16, 0, 8, 8),
                    Entity("fallingBlock", 4, 24, 0, 8, 8))
        sim = Simulator(fixture(entities, rows=("0000",)*4))
        start = SimState("test", -8, 0)
        intact = sim._mechanism_rects("test", start)
        broken = replace(start, broken=frozenset({"test:1"}), mechanisms=(
            ("test:2", 2, 7, 8.51, 0., 0.),
            ("test:3", 3, 4, 16., 0., 0.),
            ("test:4", 4, 0, 24., 0., 0.)))
        altered = sim._mechanism_rects("test", broken)
        self.assertEqual([(k, rect) for k, _, rect in altered], [("test:2", (9, 0, 17, 8))])
        subpixel = replace(broken, mechanisms=(("test:2", 2, 7, 8.49, 0., 0.),)+broken.mechanisms[1:])
        self.assertEqual(sim._mechanism_rects("test", subpixel)[0][2], (8, 0, 16, 8))
        self.assertIs(sim._mechanism_rects("test", start), intact)
        self.assertEqual(len(intact), 4)

    def test_dash_break_does_not_mutate_sibling_geometry(self):
        block = Entity("dashBlock", 1, 0, -16, 8, 32)
        sim = Simulator(fixture((block,), rows=("0000",)*4))
        start = SimState("test", -5, 0, vx=240, state=2, dash_frames=5,
                         dash_dx=1, dash_attack=12)
        intact = sim._mechanism_rects("test", start)
        moved = sim.step(start, RIGHT)
        self.assertIn("test:1", moved.broken)
        self.assertTrue(sim.solid_at("test", -3, 0, state=start))
        self.assertFalse(sim.solid_at("test", -3, 0, state=moved))
        self.assertIs(sim._mechanism_rects("test", start), intact)
        self.assertEqual(sim.step(start, RIGHT), moved)

    def test_caches_are_bounded_and_do_not_cross_maps(self):
        sim = Simulator(fixture((Entity("dashBlock", 1, 0, 0, 8, 8),)))
        other = Simulator(fixture(rows=("0000",)*4))
        self.assertTrue(sim.solid_at("test", -20, -8))
        self.assertFalse(other.solid_at("test", -20, -8))
        tile_limit = sim._tile_span_occupied.cache_info().maxsize
        for n in range(tile_limit+1):
            sim._tile_span_occupied("test", n, n, 0, 0)
        self.assertEqual(sim._tile_span_occupied.cache_info().currsize, tile_limit)
        mechanism_limit = sim._cached_mechanism_rects.cache_info().maxsize
        for n in range(mechanism_limit+1):
            sim._mechanism_rects("test", SimState("test", 0, 0,
                                mechanisms=(("test:1", 0, n, 0., 0., 0.),)))
        self.assertEqual(sim._cached_mechanism_rects.cache_info().currsize, mechanism_limit)


if __name__ == "__main__":
    unittest.main()
