"""Fast-fall underpasses preserve physical walls, hazards and caller limits."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DOWN, GRAB, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.wall_underpasses import wall_underpass_edges
from celeste_search.zip_grips import ZipContactGoal


def fixture(mirror=False, offset=(0,0), fatal_floor=False, dynamic_source=False):
    ox,oy=offset
    def x(value):return ox+(224-value if mirror else value)
    def left(value,width):return x(value+width) if mirror else x(value)
    sx,zx=left(64,16),left(96,16)
    entities=(Entity("zipMover" if dynamic_source else "solid",1,sx,oy+16,16,88,
                     {},((sx,oy+8),) if dynamic_source else ()),
              Entity("spikesLeft" if mirror else "spikesRight",2,x(80),oy+24,0,80),
              Entity("spikesDown",3,left(80,72),oy+16,72,0),
              Entity("zipMover",4,zx,oy+120,16,40,{},((zx,oy+56),)),
              Entity("spikesUp",5,zx,oy+120,16,0),
              Entity("spikesUp",6,left(40,168),oy+(120 if fatal_floor else 168),168,0))
    room=Room("underpass",ox,oy,224,192,("0"*28,)*24,entities,())
    sim=Simulator(MapData("wall-underpass",{room.name:room},Path("fixture.bin")))
    face=-1 if mirror else 1
    start=SimState(room.name,x(60),oy+82,state=1,facing=face,vy=32,dashes=1,
                   stamina=64.34881591796875,prev_input=GRAB|UP|(LEFT if mirror else RIGHT),
                   rem_x=.08320057392120361 if mirror else -.08320057392120361,
                   rem_y=.11663925647735596,
                   mechanisms=(("underpass:4",0,0,float(zx),float(oy+120),0.),))
    target=ZipContactGoal("wall_grip",room.name,x(92),oy+132,
        rect=(zx,oy+120,zx+16,oy+160),facing=face,identity="underpass:4",mechanism_kind="zipMover")
    return sim,start,target


class WallUnderpassTests(unittest.TestCase):
    def test_descends_below_spiked_overhang_before_actual_moving_side_contact(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            with self.subTest(mirror=mirror):
                sim,start,target=fixture(mirror,offset)
                stats={}
                edges=wall_underpass_edges(sim,start,target,deadline=time.monotonic()+1,
                    max_expansions=96,max_edges=4,stats=stats)
                self.assertTrue(edges)
                self.assertLessEqual(len(edges),4)
                for actions,expected,label in edges:
                    state=start;events=[];below=False
                    self.assertTrue(actions[0]&DOWN)
                    for i,mask in enumerate(actions,1):
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                        below |= state.y-11>=offset[1]+104
                        if sim.last_step_started_dash:events.append(i)
                    self.assertTrue(below)
                    self.assertEqual(state,expected)
                    self.assertTrue(target.achieved(state))
                    self.assertEqual(len(events),1)
                    # A successful 16-tick release may now be refined to 15;
                    # the same actual hazard/body replay remains authoritative.
                    self.assertGreaterEqual(events[0],16 if "refined" in label else 17)
                    self.assertTrue(label.endswith("dash_starts=1"))
                self.assertLessEqual(stats["expansions"],96)

    def test_floor_spikes_and_missing_static_contact_cannot_be_bypassed(self):
        sim,start,target=fixture(fatal_floor=True)
        self.assertEqual(wall_underpass_edges(sim,start,target,deadline=time.monotonic()+1),[])
        sim,start,target=fixture()
        for bad in (replace(start,x=start.x-8),replace(start,state=0),
                    replace(start,dead=True),replace(start,stamina=0),replace(start,facing=-1)):
            with patch.object(sim,"step",side_effect=AssertionError("Invalid static contact")):
                self.assertEqual(wall_underpass_edges(sim,bad,target,deadline=time.monotonic()+1),[])
        sim,start,target=fixture(dynamic_source=True)
        with patch.object(sim,"step",side_effect=AssertionError("Dynamic source is not static")):
            self.assertEqual(wall_underpass_edges(sim,start,target,deadline=time.monotonic()+1),[])

    def test_no_dash_and_external_route_constraints_remain_authoritative(self):
        sim,start,target=fixture()
        masks=[]
        class NoDash:
            no_dash=True
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                masks.append(mask)
                return sim.step(state,mask)
        for actions,end,_ in wall_underpass_edges(NoDash(),start,target,deadline=time.monotonic()+1):
            self.assertTrue(target.achieved(end))
            self.assertFalse(any(mask&DASH for mask in actions))
        self.assertTrue(masks)
        self.assertFalse(any(mask&DASH for mask in masks))
        class Forbidden(NoDash):
            no_dash=False
            def step(self,state,mask):return replace(sim.step(state,mask),dead=True,reason="route_limit")
        stats={}
        self.assertEqual(wall_underpass_edges(Forbidden(),start,target,deadline=time.monotonic()+1,
                                              max_expansions=8,stats=stats),[])
        self.assertEqual(stats,{"expansions":8,"simulated_frames":8})

    def test_expansion_frame_and_per_step_deadline_caps(self):
        sim,start,target=fixture()
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(wall_underpass_edges(sim,start,target,deadline=time.monotonic()-1),[])
            self.assertEqual(wall_underpass_edges(sim,start,target,deadline=time.monotonic()+1,
                stats={"expansions":3,"max_expansions":3}),[])
            step.assert_not_called()
        stats={"expansions":2}
        self.assertEqual(wall_underpass_edges(sim,start,target,deadline=time.monotonic()+1,
            max_expansions=3,max_frames=4,stats=stats),[])
        self.assertEqual(stats,{"expansions":3,"simulated_frames":4})
        clock=[0.];base=sim.step
        def delayed(state,mask):
            clock[0]+=.6
            return base(state,mask)
        stats={}
        with patch("celeste_search.wall_underpasses.time.monotonic",side_effect=lambda:clock[0]), \
                patch.object(sim,"step",side_effect=delayed):
            self.assertEqual(wall_underpass_edges(sim,start,target,deadline=1,stats=stats),[])
        self.assertEqual(stats,{"expansions":1,"simulated_frames":2})


if __name__=="__main__":
    unittest.main()
