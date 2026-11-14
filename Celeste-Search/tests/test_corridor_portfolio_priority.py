"""Real corridor recipe gets a worker slot before long unrelated families.

Controlled futures only: no worker process, offline search, or game connection.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

from celeste_search.assist import TargetPlanner
from celeste_search.assist_strategies import route_strategy_specs, strategy_spec
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState
from test_assist_portfolio import Clock, FakePool, ORDER, SPECS


def fixture(multiple=False):
    room=Room('a',0,0,320,184,('0'*40,)*23,
        (Entity('zipMover',1,64,80,24,16,{},((128.,80.),)),),((16.,160.),))
    data=MapData('corridor-portfolio',{'a':room},Path('fixture.bin'))
    state=SimState('a',16,160,on_ground=True,frame=91,stamina=53.)
    points=[dict(room='a',x=200,y=80,tile_size=8)]
    if multiple:points.append(dict(room='a',x=256,y=48,tile_size=8))
    route=dict(waypoints=points,segments=[dict(no_dash=False,no_ground=False) for _ in points])
    return data,state,route


class CorridorPortfolioPriorityTests(unittest.TestCase):
    def test_corridor_is_first_with_one_or_many_workers_on_every_retry(self):
        for workers in (1,5):
            for attempt in (0,1,7):
                for multiple in (False,True):
                    with self.subTest(workers=workers,attempt=attempt,multiple=multiple):
                        data,state,route=fixture(multiple)
                        pool,clock=FakePool(),Clock()
                        planner=TargetPlanner(workers=workers,seconds=8,pool=pool,clock=clock)
                        try:
                            planner.set_map(data)
                            planner.set_deadline(clock()+90)
                            all_specs=route_strategy_specs(route,state,data)
                            self.assertTrue(planner.request('fixture.bin',state,route,attempt,['failed-route']))
                            configs=[config for config,_ in pool.calls]
                            self.assertEqual(configs[0]['strategy'],'ordered_corridor')
                            self.assertEqual(len(configs),workers)
                            self.assertEqual(Counter(row['name'] for row in planner.specs),
                                             Counter(row['name'] for row in all_specs))
                            self.assertEqual(sum(row['name']=='ordered_corridor' for row in planner.specs),1)
                            self.assertEqual(configs[0]['start_state'],state.to_dict())
                            self.assertEqual(configs[0]['route'],route)
                            self.assertEqual(configs[0]['forbidden'],['failed-route'])
                            self.assertEqual((configs[0]['seconds'],configs[0]['expansions']),(90,25000))
                            self.assertEqual(planner.deadline,190.)
                        finally:
                            planner.close()

    def test_corridor_remains_ordered_but_is_a_distinct_portfolio_family(self):
        spec=strategy_spec('ordered_corridor')
        self.assertEqual(spec['family'],'ordered')
        self.assertEqual(spec['portfolio_family'],'corridor')
        self.assertEqual(spec['variant'],0)
        for other in ('ordered_mechanism','ordered_mechanism_capability','ordered_regional'):
            self.assertEqual(strategy_spec(other)['family'],'ordered')

    def test_corridor_failure_releases_the_only_slot_for_other_families(self):
        data,state,route=fixture()
        pool,clock=FakePool(),Clock()
        planner=TargetPlanner(workers=1,seconds=8,pool=pool,clock=clock)
        try:
            planner.set_map(data)
            planner.request('fixture.bin',state,route,0,[])
            self.assertEqual(pool.calls[0][0]['strategy'],'ordered_corridor')
            pool.finish(0)
            self.assertIsNone(planner.poll())
            self.assertEqual(len(pool.calls),2)
            self.assertNotEqual(pool.calls[1][0]['strategy'],'ordered_corridor')
            self.assertEqual(planner.stats()['active'],1)
            self.assertEqual(planner.stats()['completed'],1)
            self.assertGreater(len(planner.pending),0)
        finally:
            planner.close()

    def test_priority_still_obeys_remaining_wall_time(self):
        data,state,route=fixture()
        pool,clock=FakePool(),Clock()
        planner=TargetPlanner(workers=1,seconds=8,pool=pool,clock=clock)
        try:
            planner.set_map(data)
            planner.set_deadline(clock()+2.5)
            planner.request('fixture.bin',state,route,7,[])
            config=pool.calls[0][0]
            self.assertEqual(config['strategy'],'ordered_corridor')
            self.assertEqual(config['seconds'],2.5)
            self.assertEqual(config['expansions'],25000)
        finally:
            planner.close()

    def test_without_corridor_existing_family_order_and_retry_rotation_stay_intact(self):
        _,state,route=fixture()
        for attempt,expected in ((0,ORDER),(1,ORDER[2:]+ORDER[:2])):
            pool,clock=FakePool(),Clock()
            planner=TargetPlanner(workers=2,seconds=8,pool=pool,clock=clock)
            try:
                with patch('celeste_search.assist.route_strategy_specs',return_value=deepcopy(SPECS)):
                    planner.request('fixture.bin',state,route,attempt,[])
                self.assertEqual(planner.stats()['strategies'],expected)
                self.assertEqual([config['strategy'] for config,_ in pool.calls],expected[:2])
            finally:
                planner.close()


if __name__=='__main__':unittest.main()
