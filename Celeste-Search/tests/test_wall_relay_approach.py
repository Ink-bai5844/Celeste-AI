"""Wall setups preserve real resources and the caller's per-frame constraints."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.wall_moves import WallGripGoal, wall_approach_edges, wall_relay_approach_edges


def fixture(*, mirror=False, offset=(0,0)):
    ox, oy = offset
    width = 192
    grid = [["0"]*24 for _ in range(24)]
    for left,top,right,bottom in ((0,144,40,192),(80,96,88,192),(64,8,72,80)):
        for y in range(top//8,bottom//8):
            for x in range(left//8,right//8):
                grid[y][x] = "1"
    if mirror:
        grid = [row[::-1] for row in grid]
    def x(value):
        return ox+(width-value if mirror else value)
    room = Room("switch",ox,oy,width,192,tuple("".join(row) for row in grid),
                (Entity("spikesUp",1,x(88) if mirror else x(80),oy+96,8,0),),())
    sim = Simulator(MapData("wall-switch",{room.name:room},Path("fixture.bin")))
    source = SimState(room.name,x(8),oy+144,on_ground=True,grace=6,
                      facing=-1 if mirror else 1)
    final_rect = (x(72),oy+8,x(64),oy+80) if mirror else (x(64),oy+8,x(72),oy+80)
    goal = WallGripGoal("wall_grip",room.name,x(76),oy+88,rect=final_rect,
                        facing=1 if mirror else -1,grip_offset=80)
    lower_rect = (x(88),oy+96,x(80),oy+192) if mirror else (x(80),oy+96,x(88),oy+192)
    lower = WallGripGoal("wall_grip",room.name,x(76),oy+112,rect=lower_rect,
                         facing=-1 if mirror else 1,grip_offset=16)
    return sim,source,goal,lower


class WallRelayApproachTests(unittest.TestCase):
    def test_runup_reaches_lower_face_with_more_stamina(self):
        sim,source,_,lower = fixture()
        edges = wall_approach_edges(sim,source,lower,deadline=time.monotonic()+1,
                                   max_frames=180,max_edges=2,max_expansions=24)
        self.assertTrue(edges)
        actions,state,label = edges[0]
        self.assertIn("runup_jump_grab",label)
        self.assertGreater(state.stamina,100)
        self.assertEqual(state.dashes,1)
        self.assertLess(len(actions),80)
        self.assertFalse(any(mask & DASH for mask in actions))

    def test_two_real_grips_replay_exactly_with_mirror_and_translation(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,880))):
            with self.subTest(mirror=mirror):
                sim,source,goal,_ = fixture(mirror=mirror,offset=offset)
                stats = {}
                edges = wall_relay_approach_edges(sim,source,goal,
                    deadline=time.monotonic()+1.5,max_frames=180,
                    stats=stats,max_expansions=180,max_edges=1)
                self.assertTrue(edges)
                actions,expected,label = edges[0]
                self.assertTrue(label.startswith("wall_relay:"))
                self.assertLessEqual(len(actions),180)
                self.assertLessEqual(stats["expansions"],180)
                state = source
                faces = set()
                for mask in actions:
                    self.assertFalse(mask & DASH)
                    state = sim.step(state,mask)
                    self.assertFalse(state.dead)
                    self.assertFalse(sim.last_step_started_dash)
                    if state.state==1:
                        faces.add(state.facing)
                self.assertEqual(faces,{-1,1})
                self.assertEqual(state,expected)
                self.assertTrue(goal.achieved(state))
                self.assertEqual(state.dashes,source.dashes)

    def test_deadline_expansion_and_frame_limits_are_hard(self):
        sim,source,goal,_ = fixture()
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(wall_relay_approach_edges(sim,source,goal,deadline=time.monotonic()-1),[])
            self.assertEqual(wall_relay_approach_edges(sim,source,goal,deadline=time.monotonic()+1,
                                                      stats={"expansions":4},max_expansions=4),[])
            self.assertEqual(wall_relay_approach_edges(sim,source,goal,deadline=time.monotonic()+1,max_frames=0),[])
            step.assert_not_called()
        self.assertEqual(wall_relay_approach_edges(sim,source,goal,deadline=time.monotonic()+1,
                                                  max_frames=8,max_expansions=24),[])

    def test_dynamic_wrong_kind_and_non_higher_targets_are_not_relayed(self):
        sim,source,goal,_ = fixture()
        with patch.object(sim,"step",side_effect=AssertionError("invalid relay target")):
            for target in (replace(goal,identity="switch:99"),replace(goal,kind="surface"),
                           replace(goal,y=source.y),replace(goal,room="elsewhere")):
                self.assertEqual(wall_relay_approach_edges(sim,source,target,deadline=time.monotonic()+1),[])

    def test_caller_constraints_are_observed_at_every_step(self):
        sim,source,goal,_ = fixture()
        # A public no-ground segment rejects an actual ground step in the
        # run-up. The helper must not replace the caller's simulator with base.
        route={"waypoints":[{"room":source.room,"x":72,"y":80,"tile_size":8}],
               "segments":[{"no_dash":True,"no_ground":True}]}
        from celeste_search.assist_routes import RouteProgress
        constrained=_ConstrainedSimulator(sim,route,no_dash=True,
                                           phase=RouteProgress(0,True))
        seen=[]
        base_step=constrained.step
        def step(before,mask):
            after=base_step(before,mask);seen.append(after)
            return after
        with patch.object(constrained,"step",side_effect=step):
            edges=wall_relay_approach_edges(constrained,source,goal,
                deadline=time.monotonic()+1,max_frames=180,max_expansions=80)
        self.assertTrue(seen)
        self.assertTrue(any(state.dead and state.reason.startswith("corridor_") for state in seen))
        # Every accepted alternative still has to replay through that wrapper.
        for actions,expected,_ in edges:
            current=source
            for mask in actions:
                current=base_step(current,mask)
                self.assertFalse(current.dead)
            self.assertEqual(current,expected)


if __name__ == "__main__":
    unittest.main()
