"""Private wall contacts preserve the public landing and route restrictions."""
from pathlib import Path
import time
import unittest

from celeste_search.assist_routes import start_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator, _support_grips
from celeste_search.corridor_routes import CorridorPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.wall_moves import wall_approach_edges


class CorridorWallSetupTests(unittest.TestCase):
    def fixture(self, mirror=False):
        width = 160
        def block(kind, identity, x, y, w, h):
            return Entity(kind, identity, width-x-w if mirror else x, y, w, h)
        room = Room("shaft", 0, 0, width, 144, ("0"*20,)*16+("1"*20,)*2,
            (block("solid", 1, 80, 32, 16, 64),
             block("solid", 2, 0, 64, 8, 64),
             block("jumpThru", 3, 8, 88, 24, 0)), ())
        sim = Simulator(MapData("wall-support", {room.name: room}, Path("fixture.bin")))
        start = SimState(room.name, 60. if mirror else 100., 84., state=1,
                         facing=1 if mirror else -1, dashes=1, stamina=110)
        x = 140. if mirror else 20.
        portal = CorridorPortalGoal("surface", room.name, x, 88.,
            bounds=(x-4,87.,x+4,89.), role="support")
        route = dict(waypoints=[dict(room=room.name,x=x-4,y=80,tile_size=8)],
                     segments=[dict(no_dash=False,no_ground=False)])
        return sim, start, portal, route

    def test_existing_wall_contact_can_set_up_a_distant_support(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                sim, start, portal, route = self.fixture(mirror)
                grips = _support_grips(sim, start, portal)
                self.assertTrue(grips)
                phase = start_route_progress(route, start)
                wrapped = _ConstrainedSimulator(sim, route, phase=phase)
                edges = wall_approach_edges(wrapped, start, grips[0],
                    deadline=time.monotonic()+2, max_frames=180, max_edges=1)
                self.assertTrue(edges)
                actions, final, _ = edges[0]
                current = start
                for mask in actions:
                    current = wrapped.step(current, mask)
                    self.assertFalse(current.dead)
                self.assertEqual(current, final)
                self.assertTrue(grips[0].achieved(final))
                self.assertFalse(portal.achieved(final))

    def test_no_dash_segment_rejects_dash_underpass_setup(self):
        sim, start, portal, route = self.fixture()
        route["segments"][0]["no_dash"] = True
        grip = _support_grips(sim, start, portal)[0]
        wrapped = _ConstrainedSimulator(sim, route,
            phase=start_route_progress(route, start))
        edges = wall_approach_edges(wrapped, start, grip,
            deadline=time.monotonic()+2, max_frames=180, max_edges=1)
        self.assertFalse(edges)


if __name__ == "__main__":
    unittest.main()
