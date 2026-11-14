"""Spring flight and spiked zip transfers use uninterrupted physical states."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.corridor_routes import CorridorPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import mechanism_transfer_edges
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.zip_grips import zip_grip_edges, zip_side_grip_goals, zip_top_blocked


def fixture(mirror=False, offset=(0,0)):
    ox,oy=offset
    def point(x):return ox+(400-x if mirror else x)
    def left(x,width):return point(x+width) if mirror else point(x)
    entities=(
        Entity("spring",5,point(112),oy+192,0,0),
        Entity("jumpThru",4,left(112,8),oy+192,8,0),
        Entity("zipMover",1,left(80,16),oy+112,16,24,{},((left(136,16),oy+112),)),
        Entity("zipMover",6,left(344,24),oy+88,24,32,{},((left(344,24),oy+24),)),
        Entity("spikesUp",13,left(80,16),oy+112,16,0),
        Entity("spikesUp",14,left(344,24),oy+88,24,0),
        Entity("jumpThru",12,left(320,40),oy-16,40,0))
    room=Room("relay",ox,oy-64,400,320,("0"*50,)*40,entities,())
    sim=Simulator(MapData("zip-transfer",{room.name:room},Path("fixture.bin")))
    before=SimState(room.name,point(113),oy+191,vy=60,dashes=0)
    spring=sim.step(before,0)
    target=CorridorPortalGoal("surface",room.name,point(340),oy-16,
        bounds=(left(320,40),oy-19,left(320,40)+40,oy-13),role="support")
    grip=next(g for g in zip_side_grip_goals(sim,spring,target)
              if g.identity=="relay:1" and g.facing==(1 if mirror else -1))
    return sim,spring,grip,target


class MechanismMoveTests(unittest.TestCase):
    def replay(self,sim,origin,target,edge):
        actions,final,label=edge
        state=origin;events=0
        for mask in actions:
            state=sim.step(state,mask)
            events+=bool(sim.last_step_started_dash)
            self.assertFalse(state.dead)
        self.assertEqual(state,final)
        self.assertTrue(target.achieved(state))
        self.assertTrue(label.endswith(f"dash_starts={events}"))
        return state

    def test_spring_two_spiked_zips_and_support_continue_in_both_directions(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            sim,spring,first,target=fixture(mirror,offset)
            self.assertEqual((spring.vy,spring.dashes),(-185.,1))
            self.assertTrue(all(zip_top_blocked(e,sim.map.rooms[spring.room])
                for e in sim.map.rooms[spring.room].entities if e.kind=="zipMover"))
            approaches=mechanism_transfer_edges(sim,spring,first,deadline=time.monotonic()+2)
            self.assertTrue(approaches)
            self.assertTrue(all(not any(mask&DASH for mask in actions) for actions,_,_ in approaches))
            state=self.replay(sim,spring,first,approaches[0])
            self.assertEqual(state.dashes,1)
            first_ride=list(zip_grip_edges(sim,state,first,target,deadline=time.monotonic()+1))[-1]
            state=sim.roll(state,first_ride[0][0],len(first_ride[0]))
            self.assertEqual(state,first_ride[1])
            self.assertGreater(abs(state.lift_vx),100)
            second=next(g for g in zip_side_grip_goals(sim,state,target)
                if g.identity=="relay:6" and g.facing==(-1 if mirror else 1))
            transfers=mechanism_transfer_edges(sim,state,second,deadline=time.monotonic()+2)
            self.assertTrue(transfers)
            for edge in transfers:
                self.replay(sim,state,second,edge)
            state=transfers[0][1]
            second_ride=list(zip_grip_edges(sim,state,second,target,deadline=time.monotonic()+1))[-1]
            state=sim.roll(state,second_ride[0][0],len(second_ride[0]))
            self.assertEqual(state,second_ride[1])
            self.assertLess(state.lift_vy,-130)
            launches=mechanism_transfer_edges(sim,state,target,deadline=time.monotonic()+2)
            self.assertTrue(launches)
            for edge in launches:
                final=self.replay(sim,state,target,edge)
                self.assertTrue(final.on_ground)
                self.assertFalse(any(mask&DASH for mask in edge[0]))

    def test_contact_with_first_mover_cannot_complete_the_second_mover_target(self):
        sim,spring,_,target=fixture()
        other=next(grip for grip in zip_side_grip_goals(sim,spring,target)
                   if grip.identity=="relay:6" and grip.facing==1)
        edges=mechanism_transfer_edges(sim,spring,other,
            deadline=time.monotonic()+1,max_expansions=8,max_frames=120)
        self.assertEqual(edges,[])

    def test_no_zip_contact_cannot_manufacture_a_lift_assisted_launch(self):
        sim,spring,grip,target=fixture()
        for start in (replace(spring,state=1,x=spring.x+80),
                      replace(spring,on_ground=True,state=0),replace(spring,dead=True)):
            with patch.object(sim,"step",wraps=sim.step) as step:
                self.assertEqual(mechanism_transfer_edges(sim,start,target,
                    deadline=time.monotonic()+1),[])
                step.assert_not_called()

    def test_wrapped_constraint_failure_is_never_an_arrival(self):
        sim,spring,grip,_=fixture()
        class Wrapped:
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                return replace(sim.step(state,mask),dead=True,reason="hard_limit")
        stats={}
        self.assertEqual(mechanism_transfer_edges(Wrapped(),spring,grip,
            deadline=time.monotonic()+1,stats=stats,max_expansions=8),[])
        self.assertEqual(stats,{"expansions":8,"simulated_frames":8})

    def test_expansion_frame_and_each_step_deadline_limits_are_honored(self):
        sim,spring,grip,_=fixture()
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(mechanism_transfer_edges(sim,spring,grip,
                deadline=time.monotonic()-1),[])
            self.assertEqual(mechanism_transfer_edges(sim,spring,grip,
                deadline=time.monotonic()+1,stats={"expansions":3},max_expansions=3),[])
            step.assert_not_called()
        stats={"expansions":2}
        mechanism_transfer_edges(sim,spring,grip,deadline=time.monotonic()+1,
                                  stats=stats,max_expansions=3,max_frames=4)
        self.assertEqual(stats,{"expansions":3,"simulated_frames":4})
        clock=[0.]
        original=sim.step
        def step(state,mask):
            clock[0]+=.6
            return original(state,mask)
        with patch("celeste_search.mechanism_moves.time.monotonic",side_effect=lambda:clock[0]), \
             patch.object(sim,"step",side_effect=step) as invoked:
            self.assertEqual(mechanism_transfer_edges(sim,spring,grip,deadline=1),[])
            self.assertEqual(invoked.call_count,2)


if __name__=="__main__":
    unittest.main()
