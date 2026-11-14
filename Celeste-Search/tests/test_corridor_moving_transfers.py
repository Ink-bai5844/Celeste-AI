"""Corridor guidance must retain physical mover and underhang capabilities."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, start_route_progress, advance_route_progress, route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import DOWN, Simulator
from celeste_search.planner import SearchResult
from celeste_search.wall_underpasses import wall_underpass_edges
from test_wall_underpasses import fixture


class CorridorMovingTransferTests(unittest.TestCase):
    def test_actual_underpass_is_available_to_corridor_and_keeps_user_limits(self):
        for mirror, offset in ((False, (0, 0)), (True, (-640, -800))):
            sim, start, contact = fixture(mirror, offset)
            room = replace(sim.map.rooms[start.room], spawns=((start.x, start.y),))
            sim = Simulator(replace(sim.map, rooms={room.name: room}))
            ox, oy = offset
            x = ox + (128 if mirror else 88)
            route = normalize_route(sim.map, dict(room=start.room, x=x, y=oy+128))
            goal = _selected_goal(route["waypoints"][0])[0]
            proposal = CorridorRoute("underpass", start.room, (contact,),
                ((start.x, start.y), (contact.x, contact.y)), (), 80.)
            calls = []
            def actual(*args, **kwargs):
                calls.append(kwargs.get("continuation"))
                return wall_underpass_edges(*args, **kwargs)
            def no_generic(wrapped, origin, target, **kwargs):
                return SearchResult([], origin, False, 1, 0, "frontier_exhausted")
            with patch("celeste_search.corridor_planning.generate_corridor_routes", return_value=[proposal]), \
                 patch("celeste_search.corridor_planning.search_leg", side_effect=no_generic), \
                 patch("celeste_search.wall_transfers.static_wall_transfer_edges", return_value=[]), \
                 patch("celeste_search.wall_moves.wall_approach_edges", return_value=[]), \
                 patch("celeste_search.wall_moves.wall_relay_approach_edges", return_value=[]), \
                 patch("celeste_search.wall_underpasses.wall_underpass_edges", side_effect=actual):
                result = search_corridor_routes(sim, start, route, [goal],
                    deadline=time.monotonic()+3, max_expansions=600, max_frames=180)
                self.assertTrue(result[2], result[6])
                self.assertTrue(calls)
                self.assertTrue(all(handle is not None for handle in calls))
                self.assertTrue(result[0][0] & DOWN)
                current, phase = start, start_route_progress(route, start)
                for action in result[0]:
                    previous = current
                    current = sim.step(current, action)
                    self.assertFalse(current.dead)
                    phase = advance_route_progress(route, phase, previous, current, action)
                self.assertEqual(current, result[1])
                self.assertTrue(route_complete(route, phase))
                forbidden = dict(route, segments=[dict(no_dash=True, no_ground=False)])
                rejected = search_corridor_routes(sim, start, forbidden, [goal],
                    deadline=time.monotonic()+3, max_expansions=600, max_frames=180)
                self.assertFalse(rejected[2])


if __name__ == "__main__":
    unittest.main()
