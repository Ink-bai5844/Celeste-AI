"""Static-wall launches retain actual contact, resources and route constraints."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.corridor_routes import CorridorPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.wall_moves import WallGripGoal
from celeste_search.wall_transfers import static_wall_transfer_edges


def fixture(*, mirror=False, offset=(0,0), dynamic=False):
    ox,oy=offset
    def x(value):return ox+(224-value if mirror else value)
    def left(value,width):return x(value+width) if mirror else x(value)
    entities=(Entity("zipMover" if dynamic else "solid",1,left(40,8),oy+120,8,40,
                     {},((left(56,8),oy+120),) if dynamic else ()),
              Entity("spikesUp",2,left(40,8),oy+120,8,0),
              Entity("jumpThru",3,left(136,16),oy+88,16,0),
              Entity("solid",4,left(152,8),oy+64,8,40))
    room=Room("relay",ox,oy,224,192,("0"*28,)*24,entities,())
    sim=Simulator(MapData("static-wall-transfer",{room.name:room},Path("fixture.bin")))
    state=SimState(room.name,x(36),oy+132,vy=32,dashes=1,stamina=52,
                   state=1,facing=-1 if mirror else 1,climb_wait=6,
                   prev_input=GRAB|(LEFT if mirror else RIGHT))
    target=CorridorPortalGoal("surface",room.name,x(144),oy+88,role="support",
                             bounds=(left(136,16),oy+87,left(136,16)+16,oy+89))
    return sim,state,target


class StaticWallTransferTests(unittest.TestCase):
    def test_late_dash_lands_exactly_on_support_in_both_directions(self):
        for mirror,offset in ((False,(0,0)),(True,(-800,480))):
            with self.subTest(mirror=mirror):
                sim,start,target=fixture(mirror=mirror,offset=offset)
                stats={}
                edges=static_wall_transfer_edges(sim,start,target,
                    deadline=time.monotonic()+1,max_frames=180,max_expansions=96,
                    max_edges=3,stats=stats)
                self.assertTrue(edges)
                self.assertLessEqual(len(edges),3)
                self.assertLessEqual(stats["expansions"],96)
                for actions,expected,label in edges:
                    state=start;dash_frames=[]
                    for i,mask in enumerate(actions,1):
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                        if sim.last_step_started_dash:dash_frames.append(i)
                    self.assertEqual(state,expected)
                    self.assertTrue(target.achieved(state))
                    self.assertTrue(state.on_ground)
                    self.assertEqual(len(dash_frames),1)
                    self.assertGreaterEqual(dash_frames[0],29)
                    self.assertTrue(label.endswith("dash_starts=1"))

    def test_wall_target_requires_its_actual_grip(self):
        sim,start,surface=fixture()
        target=WallGripGoal("wall_grip",start.room,148,100,rect=(152,64,160,104),
                            facing=1,grip_offset=36)
        edges=static_wall_transfer_edges(sim,start,target,deadline=time.monotonic()+1,
                                         max_expansions=96)
        self.assertTrue(edges)
        self.assertTrue(all(target.achieved(state) for _,state,_ in edges))
        self.assertTrue(all(state.state==1 and state.facing==1 for _,state,_ in edges))

    def test_fake_contact_or_dynamic_side_cannot_enter_static_family(self):
        sim,start,target=fixture()
        for origin in (replace(start,x=24),replace(start,state=0),replace(start,facing=-1),
                       replace(start,dead=True),replace(start,stamina=0)):
            with patch.object(sim,"step",side_effect=AssertionError("No physical static contact")):
                self.assertEqual(static_wall_transfer_edges(sim,origin,target,
                    deadline=time.monotonic()+1),[])
        sim,start,target=fixture(dynamic=True)
        with patch.object(sim,"step",side_effect=AssertionError("Dynamic side is not static")):
            self.assertEqual(static_wall_transfer_edges(sim,start,target,
                deadline=time.monotonic()+1),[])

    def test_no_dash_and_caller_rejection_cannot_be_bypassed(self):
        sim,start,target=fixture()
        class Wrapped:
            no_dash=True
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                self.assert_mask(mask)
                return sim.step(state,mask)
            def assert_mask(self,mask):
                if mask&DASH:raise AssertionError("No-dash wrapper received a dash proposal")
        self.assertEqual(static_wall_transfer_edges(Wrapped(),start,target,
            deadline=time.monotonic()+1,max_expansions=96),[])
        class Rejected(Wrapped):
            no_dash=False
            def step(self,state,mask):
                return replace(sim.step(state,mask),dead=True,reason="public_route_limit")
        stats={}
        self.assertEqual(static_wall_transfer_edges(Rejected(),start,target,
            deadline=time.monotonic()+1,max_expansions=8,stats=stats),[])
        self.assertEqual(stats,{"expansions":8,"simulated_frames":8})

    def test_strict_expansion_frame_and_each_step_deadline_budgets(self):
        sim,start,target=fixture()
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(static_wall_transfer_edges(sim,start,target,deadline=time.monotonic()-1),[])
            self.assertEqual(static_wall_transfer_edges(sim,start,target,deadline=time.monotonic()+1,
                stats={"expansions":4},max_expansions=4),[])
            step.assert_not_called()
        stats={"expansions":2}
        self.assertEqual(static_wall_transfer_edges(sim,start,target,deadline=time.monotonic()+1,
            max_expansions=3,max_frames=4,stats=stats),[])
        self.assertEqual(stats,{"expansions":3,"simulated_frames":4})
        clock=[0.];base=sim.step
        def step(state,mask):
            clock[0]+=.6
            return base(state,mask)
        with patch("celeste_search.wall_transfers.time.monotonic",side_effect=lambda:clock[0]), \
             patch.object(sim,"step",side_effect=step) as invoked:
            self.assertEqual(static_wall_transfer_edges(sim,start,target,deadline=1),[])
            self.assertEqual(invoked.call_count,2)


if __name__=="__main__":
    unittest.main()
