"""Conditional scenes retain bounded discovery slots across time-sliced pages."""
from pathlib import Path
import unittest
import time
from unittest.mock import patch
import celeste_search.corridor_routes as corridors
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator
from celeste_search.planner import Goal


def context():
    room=Room('a',0,0,160,160,('0'*20,)*20,
              (Entity('dashBlock',1,72,48,16,32,dict(canDash=True)),),((24,80),))
    simulator=Simulator(MapData('fairness',{'a':room},Path('fairness.bin')))
    return simulator,SimState('a',24,80),Goal('target','a',136,80)


def route(index,conditional=False):
    condition=('a:1',) if conditional else ()
    identity=('gate' if conditional else 'ordinary')+str(index)
    return corridors.CorridorRoute(identity,'a',(),((24.,80.),(136.,80.)),
                                   ('solid:'+str(index),),112.,condition)


class ConditionalCorridorFairnessTests(unittest.TestCase):
    def setUp(self):corridors.clear_corridor_cache()

    def test_full_ordinary_batch_cannot_complete_before_conditional_scene_runs(self):
        sim,start,goal=context();clock=[0.];attempts=[]
        def discover(*args,**options):
            scenario=options['discovery_scenario'];attempts.append(scenario)
            if not scenario:
                clock[0]+=.1
                return [route(i) for i in range(24)]
            return [route(100,True)]
        with patch('celeste_search.corridor_routes.time.monotonic',side_effect=lambda:clock[0]), \
                patch('celeste_search.corridor_routes._generate_corridor_target',side_effect=discover):
            first=corridors.generate_corridor_routes(sim,start,goal,max_routes=24,deadline=.05)
            self.assertEqual(len(first),18)
            self.assertFalse(first.complete)
            handle=first.continuation
            self.assertTrue(handle.conditional_work_pending)
            corridors.clear_corridor_cache()
            second=corridors.generate_corridor_routes(sim,start,goal,max_routes=24,
                candidate_offset=first.next_offset,continuation=handle,deadline=.2)
            whole=corridors.generate_corridor_routes(sim,start,goal,max_routes=24,continuation=handle,deadline=1.)
        self.assertTrue(any(scenario for scenario in attempts))
        self.assertTrue(any(value.conditional_on for value in second))
        self.assertTrue(second.complete)
        self.assertEqual([value.id for value in whole],[value.id for value in first+second])
        self.assertEqual(len({value.id for value in whole}),24)

    def test_unavailable_conditions_release_reserved_ordinary_capacity_without_regeneration(self):
        sim,start,goal=context();clock=[0.];ordinary_calls=[];condition_calls=[]
        def discover(*args,**options):
            if options['discovery_scenario']:
                condition_calls.append(1)
                if len(condition_calls)==1:clock[0]+=.1
                return []
            ordinary_calls.append(1)
            return [route(i) for i in range(24)]
        with patch('celeste_search.corridor_routes.time.monotonic',side_effect=lambda:clock[0]), \
                patch('celeste_search.corridor_routes._generate_corridor_target',side_effect=discover):
            first=corridors.generate_corridor_routes(sim,start,goal,max_routes=24,deadline=.05)
            second=corridors.generate_corridor_routes(sim,start,goal,max_routes=24,
                candidate_offset=first.next_offset,continuation=first.continuation,deadline=.2)
        self.assertEqual(len(first),18)
        self.assertEqual(len(second),6)
        self.assertEqual(len(ordinary_calls),1)
        self.assertTrue(second.complete)
        self.assertFalse(any(value.conditional_on for value in first+second))
        self.assertEqual([value.id for value in first+second],[route(i).id for i in range(24)])

    def test_raw_inventory_cannot_fill_all_96_slots_with_ordinary_routes(self):
        sim,start,goal=context();clock=[0.]
        handle=corridors._CorridorDiscovery(sim,start,goal,4,10)
        # Three earlier lane batches have filled the ordinary portion. A
        # fourth ordinary batch must leave room for a still-unseen condition.
        handle.raw=[route(i) for i in range(72)]
        handle.known={value.id for value in handle.raw}
        handle.pool=handle.raw[:18]
        def discover(*args,**options):
            clock[0]+=.1
            return ([route(100+i,True) for i in range(6)] if options['discovery_scenario']
                    else [route(72+i) for i in range(24)])
        with patch('celeste_search.corridor_routes.time.monotonic',side_effect=lambda:clock[0]), \
                patch('celeste_search.corridor_routes._generate_corridor_target',side_effect=discover):
            handle.advance(.05)
            self.assertEqual(len(handle.raw),72)
            self.assertFalse(handle.complete)
            self.assertEqual(len(handle.pool),18)
            handle.advance(.15)
        self.assertTrue(handle.complete)
        self.assertEqual(len(handle.pool),24)
        self.assertEqual(sum(bool(value.conditional_on) for value in handle.pool),6)
        self.assertTrue(all(value.id in handle.known for value in handle.pool))
        self.assertFalse(any(route(i).id in handle.known for i in range(72,96)),
                         'Rejected raw proposals must not masquerade as retained discoveries')

    def test_real_optional_dash_shortcut_competes_with_an_existing_ordinary_bypass(self):
        sim,start,goal=context()
        page=corridors.generate_corridor_routes(sim,start,goal,max_routes=6,deadline=time.monotonic()+2)
        self.assertTrue(any(not route.conditional_on for route in page),"The intact block has a real bypass")
        self.assertTrue(any(route.conditional_on for route in page),"Breaking must also be proposed when a bypass exists")
        room=sim.map.rooms['a']
        intact=corridors._current_grid(sim,room,start,4,time.monotonic()+1)
        opened=corridors._current_grid(sim,room,start,4,time.monotonic()+1,frozenset({'a:1'}))
        for proposal in page:
            grid=opened if proposal.conditional_on else intact
            self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(proposal.points,proposal.points[1:])))
            self.assertFalse(proposal.to_dict()['physically_verified'])

    def test_initial_lane_head_counts_its_unconsumed_condition_iterator(self):
        sim,start,goal=context()
        handle=corridors._CorridorDiscovery(sim,start,goal,4,10)
        self.assertFalse(handle.jobs[0][3]['scenario'])
        self.assertTrue(handle.conditional_work_pending)
        # Once the conditional task has finished, an ordinary retry alone
        # must not reserve empty slots forever.
        handle.jobs=[]
        handle.retries.append(dict(lane=goal,key=None,scenario=frozenset(),attempts=1))
        self.assertFalse(handle.conditional_work_pending)


if __name__=='__main__':unittest.main()
