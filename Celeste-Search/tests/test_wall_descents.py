"""Descending static-wall launches are physical, geometric and bounded."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DOWN, GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_descents import _launch_heights, wall_descent_edges
from celeste_search.zip_grips import ZipContactGoal


def fixture(mirror=False, offset=(0,0), fatal_floor=False, dynamic_source=False):
    ox,oy=offset
    def x(value):return ox+(224-value if mirror else value)
    def left(value,width):return x(value+width) if mirror else x(value)
    sx,zx=left(32,16),left(96,16)
    entities=(Entity("zipMover" if dynamic_source else "solid",1,sx,oy+56,16,112,
                     {},((sx,oy+16),) if dynamic_source else ()),
              Entity("solid",2,left(64,16),oy+88,16,16),
              Entity("solid",3,left(80,8),oy+8,8,80),
              Entity("spikesUp",4,left(64,16),oy+88,16,0),
              Entity("spikesRight" if mirror else "spikesLeft",5,x(64),oy+88,0,16),
              Entity("zipMover",6,zx,oy+128,16,16,{},((left(176,16),oy+128),)),
              Entity("spikesUp",7,zx,oy+128,16,0),
              Entity("spikesDown",8,zx,oy+144,16,0),
              Entity("spikesUp",9,left(64,144),oy+(128 if fatal_floor else 184),144,0))
    room=Room("descent",ox,oy,224,192,("0"*28,)*24,entities,())
    sim=Simulator(MapData("wall-descent",{room.name:room},Path("fixture.bin")))
    face=1 if mirror else -1
    start=SimState(room.name,x(52),oy+82,state=1,facing=face,vy=32,dashes=0,
        stamina=74.16666412353516,prev_input=GRAB|(RIGHT if mirror else LEFT),climb_wait=6,
        rem_x=-.4167259931564331 if mirror else .4167259931564331,rem_y=.37523090839385986,
        mechanisms=(("descent:6",0,0,float(zx),float(oy+128),0.),))
    target=ZipContactGoal("wall_grip",room.name,x(92),oy+136,
        rect=(zx,oy+128,zx+16,oy+144),facing=-face,identity="descent:6",mechanism_kind="zipMover")
    return sim,start,target


class WallDescentTests(unittest.TestCase):
    def test_lowering_before_jump_clears_side_spikes_without_inventing_resources(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            with self.subTest(mirror=mirror):
                sim,start,target=fixture(mirror,offset)
                stats={}
                edges=wall_descent_edges(sim,start,target,deadline=time.monotonic()+1,
                    max_expansions=96,max_edges=4,stats=stats)
                self.assertTrue(edges)
                self.assertTrue(any(end.stamina==start.stamina for _,end,_ in edges))
                self.assertLessEqual(len(edges),4)
                for actions,expected,label in edges:
                    state=start;lowest=start.y
                    self.assertEqual(actions[0],GRAB|DOWN|(RIGHT if mirror else LEFT))
                    for mask in actions:
                        self.assertFalse(mask&DASH)
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                        self.assertFalse(sim.last_step_started_dash)
                        lowest=max(lowest,state.y)
                    self.assertGreater(lowest,start.y+32)
                    self.assertEqual(state,expected)
                    self.assertTrue(target.achieved(state))
                    self.assertEqual(state.dashes,0)
                    self.assertLessEqual(state.stamina,start.stamina)
                    self.assertTrue(label.endswith("dash_starts=0"))
                self.assertLessEqual(stats["expansions"],96)

    def test_launch_heights_follow_target_bounds_and_source_wall_bottom(self):
        _,start,target=fixture()
        wall=(32,56,48,168)
        heights=_launch_heights(start,target,wall)
        self.assertEqual(heights,[140,132,148,124,156])
        moved=replace(target,y=target.y+16,rect=tuple(v+16 if i%2 else v for i,v in enumerate(target.rect)))
        moved_state=replace(start,mechanisms=(("descent:6",0,0,96.,144.,0.),))
        self.assertEqual(_launch_heights(moved_state,moved,wall),[h+16 for h in heights])
        short=(32,56,48,128)
        self.assertTrue(all(h<139 for h in _launch_heights(start,target,short)))

    def test_hazards_fake_contact_and_lost_grip_reject_branches(self):
        sim,start,target=fixture(fatal_floor=True)
        self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()+1),[])
        sim,start,target=fixture()
        for bad in (replace(start,x=start.x+8),replace(start,state=0),
                    replace(start,dead=True),replace(start,stamina=0)):
            with patch.object(sim,"step",side_effect=AssertionError("No real static contact")):
                self.assertEqual(wall_descent_edges(sim,bad,target,deadline=time.monotonic()+1),[])
        with patch.object(sim,"step",return_value=replace(start,state=0)):
            self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()+1),[])
        sim,start,target=fixture(dynamic_source=True)
        with patch.object(sim,"step",side_effect=AssertionError("Dynamic is not static")):
            self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()+1),[])

    def test_no_dash_external_rejection_and_public_first_hit_apply_during_descent(self):
        sim,start,target=fixture();start=replace(start,dashes=1)
        masks=[]
        class NoDash:
            no_dash=True
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                masks.append(mask)
                return sim.step(state,mask)
        edges=wall_descent_edges(NoDash(),start,target,deadline=time.monotonic()+1)
        self.assertTrue(edges)
        self.assertFalse(any(mask&DASH for mask in masks))
        class Rejected(NoDash):
            def step(self,state,mask):return replace(sim.step(state,mask),dead=True,reason="route_limit")
        stats={}
        self.assertEqual(wall_descent_edges(Rejected(),start,target,deadline=time.monotonic()+1,stats=stats),[])
        self.assertEqual(stats,{"expansions":1,"simulated_frames":1})
        class First(Goal):
            def achieved(self,state):return not state.dead and state.y>=start.y+10
        public=First("point",start.room,start.x,start.y+48)
        edges=wall_descent_edges(sim,start,public,deadline=time.monotonic()+1)
        self.assertEqual(len(edges),1)
        state=start
        for i,mask in enumerate(edges[0][0]):
            self.assertFalse(public.achieved(state))
            state=sim.step(state,mask)
        self.assertTrue(public.achieved(state))
        self.assertEqual(state,edges[0][1])

    def test_expansion_frame_and_every_step_deadline_budgets(self):
        sim,start,target=fixture()
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()-1),[])
            self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()+1,
                stats={"expansions":3},max_expansions=3),[])
            step.assert_not_called()
        stats={"expansions":2}
        self.assertEqual(wall_descent_edges(sim,start,target,deadline=time.monotonic()+1,
            max_expansions=3,max_frames=4,stats=stats),[])
        self.assertEqual(stats,{"expansions":3,"simulated_frames":4})
        clock=[0.];base=sim.step
        def delayed(state,mask):
            clock[0]+=.6
            return base(state,mask)
        stats={}
        with patch("celeste_search.wall_descents.time.monotonic",side_effect=lambda:clock[0]), \
                patch.object(sim,"step",side_effect=delayed):
            self.assertEqual(wall_descent_edges(sim,start,target,deadline=1,stats=stats),[])
        self.assertEqual(stats,{"expansions":1,"simulated_frames":2})


if __name__=="__main__":
    unittest.main()
