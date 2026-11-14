"""The native chapter-wide SolidTiles collider continues across room borders."""
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, UP, LEFT, RIGHT, SimState, Simulator


def fixture(*, mirror=False, ox=0, oy=0, upper_wall=True):
    wall_col = 2 if mirror else 5
    row = ''.join('1' if x == wall_col else '0' for x in range(8))
    upper_rows = (row,)*8 if upper_wall else ('0'*8,)*8
    upper = Room('upper', ox, oy-64, 64, 64, upper_rows, (), ())
    lower = Room('lower', ox, oy, 64, 64, (row,)*8, (), ((ox+32, oy+48),))
    # This entity is deliberately not a global static tile. Unloaded room
    # movers, barriers and one-way entities must not enter border queries.
    remote = Room('remote', ox+256, oy, 64, 64, ('0'*8,)*8,
                  (Entity('solid', 1, ox+260, oy+16, 16, 16),), ())
    sim = Simulator(MapData('global-solid-grid', {r.name:r for r in (lower, upper, remote)},
                            Path('fixture.bin')))
    facing = 1 if mirror else -1
    state = SimState('lower', ox+(12 if mirror else 52), oy+6,
                     state=1, facing=facing, vy=-45, stamina=60,
                     prev_input=GRAB|UP|(RIGHT if mirror else LEFT))
    return sim, state


class NeighborSolidTilesTests(unittest.TestCase):
    def test_neighbor_hands_do_not_invent_ledge_hop(self):
        for mirror, ox, oy in ((False, 0, 0), (True, -512, -256)):
            sim, state = fixture(mirror=mirror, ox=ox, oy=oy)
            result = sim.step(state, state.prev_input)
            self.assertEqual(result.room, 'lower')
            self.assertEqual(result.state, 1)
            self.assertEqual(result.y, oy+5)
            self.assertEqual(result.vy, -45)
            self.assertLess(result.stamina, state.stamina)

    def test_real_open_neighbor_still_allows_ledge_hop(self):
        sim, state = fixture(upper_wall=False)
        result = sim.step(state, state.prev_input)
        self.assertEqual(result.state, 0)
        self.assertEqual(result.vy, -120)

    def test_global_tiles_are_collision_only_without_loading_neighbor_entities(self):
        sim, state = fixture()
        self.assertNotIn((5, -1), sim._tiles['lower'])
        self.assertTrue(sim._tile_span_occupied('lower', 5, 5, -1, -1))
        self.assertTrue(sim.solid_at('lower', 44, -2))
        self.assertFalse(sim.solid_at('lower', 268, 28))
        self.assertFalse(sim.solid_at('lower', 28, -2))
        self.assertFalse(sim._mechanism_rects('lower', state))
        self.assertTrue(all(t >= 0 for _,t,_,_ in sim._static_rects['lower']))

    def test_same_room_collision_does_not_query_chapter_grid(self):
        class NoGlobalQuery:
            def __contains__(self, value):
                raise AssertionError('in-room lookup should use the local fast path')
        sim, _ = fixture()
        sim._world_tiles = NoGlobalQuery()
        self.assertTrue(sim._tile_span_occupied('lower', 5, 5, 2, 2))
        self.assertFalse(sim._tile_span_occupied('lower', 3, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
