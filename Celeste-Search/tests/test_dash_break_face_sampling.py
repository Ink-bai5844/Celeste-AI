"""One blocked guide point must not erase an otherwise usable impact face."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest
import time

from celeste_search.maps import Entity,MapData,Room
from celeste_search.mechanism_routes import dash_break_goals,_clear,_inventory
from celeste_search.physics import SimState,Simulator,DASH
from celeste_search.mechanism_moves import active_dash_block_edges
from celeste_search.planner import Goal


def fixture(entities, *, width=192,height=160,offset=(0,0)):
    ox,oy=offset
    entities=tuple(replace(e,x=e.x+ox,y=e.y+oy) for e in entities)
    room=Room('faces',ox,oy,width,height,('0'*(width//8),)*(height//8),entities,())
    sim=Simulator(MapData('face-sampling',{'faces':room},Path('face-sampling.bin')))
    return sim,SimState('faces',24+ox,24+oy,dashes=1)


def block(identifier=1,x=72,y=48,width=16,height=32,**attrs):
    return Entity('dashBlock',identifier,x,y,width,height,dict(canDash=True)|attrs)


class DashBreakFaceSamplingTests(unittest.TestCase):
    def test_side_under_overhang_is_retained_and_translation_invariant(self):
        for offset in ((0,0),(-704,-1200)):
            with self.subTest(offset=offset):
                sim,start=fixture((block(),Entity('solid',2,56,16,32,40)),offset=offset)
                ox,oy=offset;before=start.to_dict()
                self.assertFalse(_clear(sim,start,64+ox,52+oy),'Old single clamped height is obstructed')
                with patch.object(sim,'step',side_effect=AssertionError('Geometry cannot simulate or relocate')):
                    goals=dash_break_goals(sim,start)
                goal=next(g for g in goals if g.dash_dx==1)
                self.assertEqual((goal.x,goal.y),(64+ox,67+oy))
                self.assertTrue(_clear(sim,start,goal.x,goal.y))
                self.assertEqual(len(goals),len({(g.identity,g.dash_dx,g.dash_dy) for g in goals}))
                self.assertEqual(start.to_dict(),before)

    def test_vertical_face_searches_other_width_when_clamped_position_is_obstructed(self):
        sim,start=fixture((block(x=48,y=72,width=32,height=16),Entity('solid',2,16,56,40,32)))
        self.assertFalse(_clear(sim,start,52,68))
        goal=next(g for g in dash_break_goals(sim,start) if g.dash_dy==1)
        self.assertEqual((goal.x,goal.y),(60,68))
        self.assertTrue(_clear(sim,start,goal.x,goal.y))

    def test_long_face_preserves_a_thin_opening_between_uniform_samples(self):
        sim,start=fixture((block(y=32,height=4000),
            Entity('solid',2,56,0,16,1777),Entity('solid',3,56,1789,16,2307)),height=4096)
        with patch.object(sim,'solid_at',wraps=sim.solid_at) as solid:
            goals=dash_break_goals(sim,start)
        goal=next(g for g in goals if g.dash_dx==1)
        self.assertEqual((goal.x,goal.y),(64,1788))
        self.assertLess(solid.call_count,220,'Long blocks must not require thousands of per-pixel queries')
        self.assertTrue(_clear(sim,start,goal.x,goal.y))

    def test_hazard_rejection_tries_another_contact_coordinate(self):
        sim,start=fixture((block(),Entity('spinner',2,64,64,0,0)))
        start=replace(start,y=64)
        blocked_pose=replace(start,x=64,y=64,vx=1,vy=0,ducking=False)
        self.assertTrue(sim._hazard_hit(blocked_pose))
        goals=dash_break_goals(sim,start)
        left=next(g for g in goals if g.dash_dx==1)
        self.assertNotEqual(left.y,64)
        for goal in goals:
            pose=replace(start,x=goal.x,y=goal.y,vx=goal.dash_dx,vy=goal.dash_dy,ducking=False)
            self.assertFalse(sim._hazard_hit(pose))
            self.assertTrue(_clear(sim,start,goal.x,goal.y))

    def test_faces_and_distinct_blocks_keep_existing_inventory_quotas(self):
        sim,start=fixture((block(1,x=64),block(2,x=128)))
        goals=dash_break_goals(sim,start)
        self.assertEqual({g.identity for g in goals},{'faces:1','faces:2'})
        self.assertEqual(len(goals),8)
        self.assertEqual(len({(g.identity,g.dash_dx,g.dash_dy) for g in goals}),8)
        inventory=[node for node in _inventory(sim,start,Goal('target','faces',176,48)) if node.family=='dash_break']
        self.assertLessEqual(len(inventory),4)
        self.assertEqual({node.identity for node in inventory},{'faces:1','faces:2'})

    def test_vertical_edge_contact_has_real_approach_and_break_replay(self):
        # An eight-pixel shaft constrains the real player to x=93. Its body
        # overlaps the block at x=96 by one pixel; center-inside bounds would
        # reject this valid downward collision even after the real fall.
        sim,start=fixture((block(x=96,y=160,width=16,height=32),
            Entity('solid',2,81,0,8,152),Entity('solid',3,97,0,8,152)),height=224)
        start=replace(start,x=93,y=80,frame=209,rem_y=.125)
        target=next(g for g in dash_break_goals(sim,start) if g.dash_dy==1)
        self.assertEqual((target.x,target.y),(93,156))
        self.assertLess(target.x,target.rect[0]+1,'This guide was outside the old center-only acceptance')
        rows=active_dash_block_edges(sim,start,target,deadline=time.monotonic()+1,
                                    max_frames=180,max_expansions=320,max_edges=2)
        self.assertTrue(rows)
        for actions,final,_ in rows:
            self.assertGreater(len(actions),10)
            first_dash=next(i for i,mask in enumerate(actions) if mask&DASH)
            self.assertGreater(first_dash,0,'The complete physically falling prefix must be retained')
            actual=start
            for mask in actions:
                actual=sim.step(actual,mask)
                self.assertFalse(actual.dead)
            self.assertEqual(actual,final)
            self.assertIn('faces:1',final.broken)
        self.assertEqual(start.broken,frozenset())

    def test_closed_faces_broken_and_unbreakable_blocks_are_not_invented(self):
        sim,start=fixture((block(),Entity('solid',2,48,24,64,96)))
        self.assertEqual(dash_break_goals(sim,start),[])
        for can_dash in (False,'false',0):
            sim,start=fixture((block(canDash=can_dash),))
            self.assertEqual(dash_break_goals(sim,start),[])
        sim,start=fixture((block(),))
        self.assertEqual(dash_break_goals(sim,replace(start,broken=frozenset({'faces:1'}))),[])


if __name__=='__main__':unittest.main()
