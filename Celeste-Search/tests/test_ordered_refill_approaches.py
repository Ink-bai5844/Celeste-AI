"""Continuous refill-assisted round trips on independent synthetic geometry."""
from pathlib import Path
from unittest.mock import patch
import unittest

from celeste_search.assist_planning import plan_to_route
from celeste_search.assist_routes import (advance_route_progress, normalize_route,
    route_complete, start_route_progress)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.routes import snapshot


def fixture(mirror=False, offset=(0, 0)):
    ox, oy = offset
    point = lambda x: ox + (320-x if mirror else x)
    rect = lambda x, width: ox + (320-x-width if mirror else x)
    entities = (Entity('solid', 1, rect(0, 104), oy+192, 104, 64),
                Entity('solid', 2, rect(256, 64), oy+192, 64, 64),
                Entity('refill', 3, point(176), oy+160, 0, 0))
    room = Room('gap', ox, oy, 320, 256, ('0'*40,)*32, entities,
                ((point(24), oy+192.),))
    data = MapData('round-trip', {'gap': room}, Path('fixture.bin'))
    start = SimState('gap', point(24), oy+192, on_ground=True, grace=6,
                     frame=741, rem_x=-.125 if mirror else .125)
    route = normalize_route(data, {'waypoints': [
        dict(room='gap', x=rect(x, 8), y=oy+184) for x in (280, 24)]})
    return data, start, route


class OrderedRefillApproachesTests(unittest.TestCase):
    def test_round_trip_search_and_exact_replay_mirror_and_translate(self):
        for mirror, offset in ((False, (0, 0)), (True, (-640, -912))):
            with self.subTest(mirror=mirror):
                data, start, route = fixture(mirror, offset)
                original = start.to_dict()
                with patch('celeste_search.assist_planning._context',
                           return_value=(data, Simulator(data))):
                    result = plan_to_route(str(data.source), original, route,
                        strategy='ordered_semantic', seconds=6,
                        expansions=4000, max_attempts=1)
                self.assertTrue(result['can_execute'], result)
                self.assertEqual(result['route'], route)
                self.assertEqual(start.to_dict(), original)
                self.assertTrue(any(stage['family']=='refill_approach' and stage['reached']
                    for stage in result['strategy_details']['zip_stages']))
                sim, state = Simulator(data), start
                phase = start_route_progress(route, start)
                hits = []
                for index, mask in enumerate(result['actions'], 1):
                    after = sim.step(state, mask)
                    following = advance_route_progress(route, phase, state, after, mask)
                    self.assertFalse(after.dead)
                    self.assertIsNone(following.violation)
                    self.assertEqual(snapshot(after), result['expected'][index])
                    if following.index != phase.index:
                        hits.append(index)
                    state, phase = after, following
                self.assertEqual(len(hits), 2)
                self.assertTrue(route_complete(route, phase))
                self.assertEqual(state, SimState.from_dict(result['final_state']))

    def test_restricted_route_never_inserts_unrestricted_refill_proposals(self):
        data, start, route = fixture()
        route['segments'][0]['no_dash'] = True
        with patch('celeste_search.assist_planning._context',
                   return_value=(data, Simulator(data))), patch(
                'celeste_search.refill_moves.refill_approach_edges',
                side_effect=AssertionError('restricted route must retain its monitor')):
            result = plan_to_route(str(data.source), start.to_dict(), route,
                strategy='ordered_semantic', seconds=.2, expansions=30, max_attempts=1)
        self.assertFalse(result['can_execute'])
        self.assertEqual(result['route'], route)


if __name__ == '__main__':
    unittest.main()
