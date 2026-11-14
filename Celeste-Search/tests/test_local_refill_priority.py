"""A local refill is an early physical relay proposal, never a resource grant."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest

from celeste_search.assist_routes import normalize_route
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import Simulator,SimState
from celeste_search.planner import Goal
from celeste_search.mechanism_routes import relay_candidate_details
from celeste_search.mechanism_moves import active_dash_block_edges


def fixture(*, refill_x=144, cooldown=0, mirror=False, offset=(0,0), mover=False, second=False):
    ox,oy=offset
    point=lambda x:ox+(320-x if mirror else x)
    rect=lambda x,w:ox+(320-x-w if mirror else x)
    es=[Entity('solid',1,rect(80,8),oy+32,8,144),
        Entity('solid',2,rect(40,8),oy+64,8,112),
        Entity('refill',3,point(refill_x),oy+96,0,0),
        Entity('dashBlock',4,rect(192,16),oy+112,16,48,attrs={'canDash':True})]
    if mover:
        es.append(Entity('zipMover',5,rect(112,16),oy+80,16,24,{},((rect(160,16),oy+80),)))
    if second:es.append(Entity('refill',6,point(136),oy+96,0,0))
    r=Room('local',ox,oy,320,192,('0'*40,)*22+('1'*40,)*2,tuple(es),((point(76),oy+176.),))
    sim=Simulator(MapData('local-refill',{'local':r},Path('fixture.bin')))
    s=SimState('local',point(76),oy+100,state=1,facing=-1 if mirror else 1,
        on_ground=False,dashes=1,stamina=91,frame=123,rem_x=.125,
        refill_timers=(('local:3',cooldown),),
        mechanisms=tuple((f'local:{e.id}',0,0,e.x,e.y,0.) for e in es if e.kind=='zipMover'))
    g=Goal('target','local',point(280),oy+80)
    return sim,s,g


class LocalRefillPriorityTests(unittest.TestCase):
    def test_wall_full_dash_forward_refill_first_preserves_return_wall_and_block(self):
        for mirror,offset in ((False,(0,0)),(True,(-704,-1200))):
            with self.subTest(mirror=mirror):
                sim,state,goal=fixture(mirror=mirror,offset=offset)
                before=state.to_dict()
                with patch.object(sim,'step',side_effect=AssertionError('ranking cannot simulate')):
                    rows=relay_candidate_details(sim,state,goal,limit=6)
                self.assertEqual(rows[0]['identity'],'local:3')
                self.assertEqual(rows[0]['family'],'refill')
                self.assertEqual(rows[1]['family'],'wall')
                self.assertTrue(any(r['family']=='dash_break' for r in rows))
                self.assertEqual(len(rows),6)
                self.assertFalse(rows[0]['goal'].achieved(state))
                self.assertEqual(state.to_dict(),before)
                visited=relay_candidate_details(sim,state,goal,visited=[rows[0]['goal'].key],limit=6)
                self.assertNotIn(rows[0]['goal'],[r['goal'] for r in visited])

    def test_cold_far_and_backward_refills_do_not_displace_existing_first_probe(self):
        for kwargs in (dict(cooldown=1),dict(refill_x=200),dict(refill_x=8)):
            with self.subTest(kwargs=kwargs):
                sim,state,goal=fixture(**kwargs)
                rows=relay_candidate_details(sim,state,goal,limit=6)
                self.assertEqual(rows[0]['family'],'wall')
                self.assertTrue(any(r['family']=='dash_break' for r in rows))

    def test_only_one_refill_slot_and_existing_near_mover_stays_first(self):
        sim,state,goal=fixture(mover=True,second=True)
        rows=relay_candidate_details(sim,state,goal,limit=8)
        self.assertEqual(rows[0]['family'],'zip')
        self.assertEqual(rows[1]['identity'],'local:6')
        self.assertEqual(rows[2]['family'],'wall')
        self.assertLessEqual(len(rows),8)
        self.assertTrue(any(r['family']=='dash_break' for r in rows))
        self.assertEqual(relay_candidate_details(sim,state,goal,limit=1)[0]['family'],'zip')

    def test_ground_uses_112_pixel_radius_wall_uses_96(self):
        sim,state,goal=fixture(refill_x=180)
        self.assertEqual(relay_candidate_details(sim,state,goal,limit=6)[0]['family'],'wall')
        grounded=replace(state,state=0,on_ground=True,vy=0)
        self.assertEqual(relay_candidate_details(sim,grounded,goal,limit=6)[0]['family'],'refill')

    def test_exposed_no_dash_constraint_keeps_old_order_and_no_action_override(self):
        sim,state,goal=fixture()
        route=normalize_route(sim.map,dict(room=state.room,x=goal.x,y=goal.y))
        route['segments'][0]['no_dash']=True
        wrapped=_ConstrainedSimulator(sim,route,no_dash=True)
        rows=relay_candidate_details(wrapped,state,goal,limit=6)
        self.assertEqual(rows[0]['family'],'wall')
        target=next(row['goal'] for row in rows if row['family']=='dash_break')
        import time
        with patch.object(sim,'step',side_effect=AssertionError('hard no-dash cannot fly or break')):
            self.assertEqual(active_dash_block_edges(wrapped,state,target,deadline=time.monotonic()+1),[])

    def test_airborne_origin_does_not_gain_the_contact_promotion(self):
        sim,state,goal=fixture()
        rows=relay_candidate_details(sim,state,goal,limit=24)
        refill=next(row for row in rows if row['family']=='refill')
        wall=next(row for row in rows if row['family']=='wall')
        with patch('celeste_search.mechanism_routes._rank',return_value=[wall,refill]),\
             patch('celeste_search.mechanism_routes._inventory',return_value=[]):
            selected=relay_candidate_details(sim,replace(state,state=0,on_ground=False),goal,limit=2)
        self.assertEqual([row['family'] for row in selected],['wall','refill'])


if __name__=='__main__':unittest.main()
