"""Airborne approaches keep real flight, landing, recovery and collision state."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import normalize_route, RouteProgress
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import active_dash_block_edges, _airborne_break_approach_recipes
from celeste_search.mechanism_routes import dash_break_goals
from celeste_search.physics import Simulator, DASH
from celeste_search.planner import SearchResult


def fixture(*, mirror=False, offset=(0,0), hazard=False):
    ox,oy=offset
    point=lambda x: ox+(256-x if mirror else x)
    rect=lambda x,w: ox+(256-x-w if mirror else x)
    entities=[Entity('dashBlock',1,rect(168,16),oy+112,16,24,attrs={'canDash':True}),
              Entity('jumpThru',2,rect(128,56),oy+136,56,8)]
    if hazard:
        entities.append(Entity('spikesUp',3,rect(128,56),oy+136,56,8))
    room=Room('air',ox,oy,256,192,('0'*32,)*24,tuple(entities),((point(32),oy+48.),))
    sim=Simulator(MapData('airborne-break',{'air':room},Path('airborne-break.bin')))
    start=replace(sim.initial_state(),vx=-90 if mirror else 90,vy=160,
        on_ground=False,frame=201,rem_x=-.125 if mirror else .125,rem_y=.0625,
        facing=-1 if mirror else 1,refill_timers=(('old:refill',37),))
    target=next(g for g in dash_break_goals(sim,start) if g.dash_dx==(-1 if mirror else 1))
    route=normalize_route(sim.map,dict(room='air',x=point(216),y=oy+128))
    return sim,start,target,route


def no_search(state):
    return patch('celeste_search.planner.search_leg',
        return_value=SearchResult([],state,False,1,0,'frontier_exhausted'))


class AirborneBreakTests(unittest.TestCase):
    def test_real_two_dash_flight_landing_recovery_and_break_mirror_translation(self):
        for mirror,offset in ((False,(0,0)),(True,(-704,-1200))):
            with self.subTest(mirror=mirror):
                sim,start,target,_=fixture(mirror=mirror,offset=offset)
                before=start.to_dict();stats={}
                with no_search(start):
                    rows=active_dash_block_edges(sim,start,target,
                        deadline=time.monotonic()+1,max_edges=1,stats=stats)
                self.assertTrue(rows)
                actions,final,label=rows[0]
                self.assertIn('via_air_coast_dash',label)
                state=start;dashes=[];landings=[];recoveries=[]
                for index,mask in enumerate(actions):
                    previous=state;state=sim.step(state,mask)
                    self.assertFalse(state.dead)
                    if sim.last_step_started_dash:dashes.append(index)
                    if state.on_ground and not previous.on_ground:landings.append(index)
                    if state.dashes>previous.dashes:recoveries.append(index)
                self.assertEqual(state,final)
                self.assertEqual(len(dashes),2)
                self.assertTrue(any(dashes[0]<i<dashes[1] for i in landings))
                self.assertTrue(any(dashes[0]<i<=dashes[1] for i in recoveries))
                self.assertIn(target.identity,final.broken)
                self.assertEqual(start.to_dict(),before)
                self.assertLessEqual(stats['expansions'],320)
                self.assertLessEqual(len(actions),220)

    def test_route_restrictions_are_enforced_on_every_real_step(self):
        for restriction in ('no_dash','no_ground'):
            with self.subTest(restriction=restriction):
                sim,start,target,route=fixture()
                route['segments'][0][restriction]=True
                wrapped=_ConstrainedSimulator(sim,route,no_dash=restriction=='no_dash',
                    phase=RouteProgress(0,True))
                with no_search(start):
                    rows=active_dash_block_edges(wrapped,start,target,
                        deadline=time.monotonic()+1,max_edges=1)
                self.assertEqual(rows,[])
                self.assertFalse(start.broken)

    def test_hazard_landing_and_frame_budget_cannot_be_accepted_as_break(self):
        for hazard,frames in ((True,220),(False,10)):
            with self.subTest(hazard=hazard,frames=frames):
                sim,start,target,_=fixture(hazard=hazard)
                with no_search(start):
                    rows=active_dash_block_edges(sim,start,target,
                        deadline=time.monotonic()+1,max_frames=frames,max_edges=1)
                self.assertEqual(rows,[])

    def test_recipe_count_and_launch_envelope_are_bounded(self):
        _,start,target,_=fixture()
        recipes=list(_airborne_break_approach_recipes(start,target))
        self.assertLessEqual(len(recipes),32)
        self.assertTrue(recipes)
        self.assertTrue(all(not mask&DASH for recipe,_ in recipes[:2] for mask in recipe))
        self.assertTrue(all(len(recipe)<=140 for recipe,_ in recipes))
        for state in (replace(start,on_ground=True),replace(start,state=1),
                      replace(start,room='other'),replace(start,x=target.x+193),
                      replace(start,y=target.y+161)):
            self.assertEqual(list(_airborne_break_approach_recipes(state,target)),[])
        no_dash=list(_airborne_break_approach_recipes(replace(start,dashes=0),target))
        self.assertTrue(all(not mask&DASH for recipe,_ in no_dash for mask in recipe))
        held=list(_airborne_break_approach_recipes(replace(start,prev_input=DASH),target))
        self.assertTrue(all(not recipe[0]&DASH for recipe,_ in held))

    def test_expired_and_small_expansion_budgets_do_not_start_extra_recipes(self):
        sim,start,target,_=fixture();stats={'expansions':20}
        with patch.object(sim,'step',side_effect=AssertionError('expired work')):
            self.assertEqual(active_dash_block_edges(sim,start,target,
                deadline=time.monotonic()-1,stats=stats),[])
        with no_search(start):
            rows=active_dash_block_edges(sim,start,target,
                deadline=time.monotonic()+1,stats=stats,max_expansions=24,max_edges=1)
        self.assertEqual(rows,[])
        self.assertLessEqual(stats['expansions'],24)


if __name__=='__main__':unittest.main()
