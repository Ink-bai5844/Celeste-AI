"""Opt-in relay samples follow one real return and subsequent launch."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import WallGripGoal
from celeste_search.zip_grips import _contact, zip_grip_edges


def fixture(mirror=False, offset=(0, 0)):
    ox, oy = offset
    x = ox + (224 if mirror else 80)
    mover = Entity("zipMover", 1, x, oy+128, 16, 40, {}, ((x, oy+64),))
    room = Room("cycle", ox, oy, 320, 320, ("0"*40,)*40,
                (mover, Entity("spikesUp", 2, x, oy+128, 16, 0)), ())
    sim = Simulator(MapData("zip-cycle", {room.name:room}, Path("fixture.bin")))
    facing = 1 if mirror else -1
    px = x-4 if mirror else x+20
    grip = WallGripGoal("wall_grip", room.name, px, oy+140, rect=(x,oy+128,x+16,oy+168),
                       identity="cycle:1", mechanism_kind="zipMover", facing=facing)
    mask = GRAB | (RIGHT if facing > 0 else LEFT)
    start = SimState(room.name, px, oy+154, state=1, facing=facing, stamina=110,
                     dashes=1, climb_wait=5, prev_input=mask,
                     mechanisms=(("cycle:1",1,7,float(x),float(oy+128),0.),))
    goal = Goal("room", "unreached", 0, 0)
    return sim, start, grip, goal, mask


class ZipCycleWindowTests(unittest.TestCase):
    def test_return_phase_exposes_real_renewed_lift_windows_with_exact_replay(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            with self.subTest(mirror=mirror):
                sim,start,grip,goal,mask=fixture(mirror,offset)
                for _ in range(116):
                    start=sim.step(start,mask)
                self.assertEqual(start.mechanisms[0][1:3],(4,48))
                stats={}
                edges=list(zip_grip_edges(sim,start,grip,goal,cycle_windows=True,
                    max_frames=220,deadline=time.monotonic()+1,stats=stats))
                outgoing={label.rsplit(":",1)[-1]: (len(actions),state)
                          for actions,state,label in edges if ":outgoing" in label}
                self.assertEqual({key:value[0] for key,value in outgoing.items()},
                                 {"outgoing8":118,"outgoing16":126,"outgoing24":134})
                self.assertTrue(all(state.lift_vy<0 for _,state in outgoing.values()))
                self.assertEqual(len(edges[-1][0]),140)
                self.assertTrue(edges[-1][2].endswith(":ride"))
                self.assertEqual(stats,{"expansions":1,"simulated_frames":140})
                self.assertEqual(len({len(row[0]) for row in edges}),len(edges))
                self.assertLessEqual(len(edges),14)
                for actions,end,_ in edges:
                    check=start
                    for action in actions:
                        self.assertEqual(action,mask)
                        check=sim.step(check,action)
                        self.assertTrue(_contact(check,grip))
                    self.assertEqual(check,end)
                    self.assertLess(end.stamina,start.stamina)
                    self.assertEqual(end.dashes,start.dashes)

    def test_starting_at_endpoint_waits_one_cycle_and_default_still_stops_immediately(self):
        sim,start,grip,goal,mask=fixture()
        for _ in range(37):
            start=sim.step(start,mask)
        self.assertEqual(start.mechanisms[0][1:3],(3,31))
        default=list(zip_grip_edges(sim,start,grip,goal,max_frames=500,
                                   deadline=time.monotonic()+1))
        self.assertEqual([len(actions) for actions,_,_ in default],[1])
        stats={}
        cycle=list(zip_grip_edges(sim,start,grip,goal,max_frames=500,cycle_windows=True,
                                 deadline=time.monotonic()+1,stats=stats))
        self.assertEqual(len(cycle[-1][0]),219)
        self.assertEqual(cycle[-1][1].mechanisms[0][1:3],(3,31))
        self.assertEqual(stats["simulated_frames"],219)
        self.assertEqual(sum(label.endswith(":ride") for _,_,label in cycle),1)
        self.assertLessEqual(len(cycle),20)

    def test_frame_expansion_and_every_step_deadline_caps_are_authoritative(self):
        sim,start,grip,goal,mask=fixture()
        for _ in range(116):start=sim.step(start,mask)
        stats={"expansions":3,"simulated_frames":10,"max_expansions":4}
        edges=list(zip_grip_edges(sim,start,grip,goal,max_frames=100,cycle_windows=True,
                                 deadline=time.monotonic()+1,stats=stats))
        self.assertEqual(stats,{"expansions":4,"simulated_frames":110,"max_expansions":4})
        self.assertEqual([len(row[0]) for row in edges],[8,16,24,32,48,72])
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(list(zip_grip_edges(sim,start,grip,goal,cycle_windows=True,
                deadline=time.monotonic()+1,stats=stats)),[])
            step.assert_not_called()
        clock=[0.]
        real_step=sim.step
        def delayed(state,action):
            clock[0]+=.6
            return real_step(state,action)
        counted={}
        with patch("celeste_search.zip_grips.time.monotonic",side_effect=lambda:clock[0]), \
                patch.object(sim,"step",side_effect=delayed):
            self.assertEqual(list(zip_grip_edges(sim,start,grip,goal,cycle_windows=True,
                deadline=1.,stats=counted)),[])
        self.assertEqual(counted,{"expansions":1,"simulated_frames":2})

    def test_lost_contact_or_constraint_stops_cycle_and_public_first_hit_is_kept(self):
        sim,start,grip,goal,_=fixture()
        for changed in ({"state":0},{"dead":True},{"stamina":0}):
            with self.subTest(changed=changed), patch.object(sim,"step",return_value=replace(start,**changed)):
                self.assertEqual(list(zip_grip_edges(sim,start,grip,goal,cycle_windows=True,
                    deadline=time.monotonic()+1)),[])
        crossing=replace(start,room="unreached",state=0,frame=start.frame+1)
        with patch.object(sim,"step",return_value=crossing):
            edges=list(zip_grip_edges(sim,start,grip,goal,cycle_windows=True,
                                     deadline=time.monotonic()+1))
        self.assertEqual(len(edges),1)
        self.assertEqual(edges[0][1],crossing)
        self.assertTrue(edges[0][2].endswith(":goal"))


if __name__=="__main__":
    unittest.main()
