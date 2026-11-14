"""Destructible passage macros preserve real collisions across time slices."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import RouteProgress
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.corridor_planning import _StageGoal
from celeste_search.corridor_routes import DashBlockBreakGoal
from celeste_search.dash_block_moves import dash_block_edges
from celeste_search.macro_continuation import MacroContinuation
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, RIGHT, Simulator, SimState
from celeste_search.planner import Goal


def fixture(direction=(1,0), *, can_dash=True, identity="plain:7"):
    block=Entity("dashBlock",7,88,80,16,16,attrs={"canDash":can_dash})
    room=Room("plain",0,0,192,192,("0"*24,)*22+("1"*24,)*2,(block,),((68.,92.),))
    sim=Simulator(MapData("dash-block",{"plain":room},Path("dash-block.bin")))
    x,y={(1,0):(68.,92.),(-1,0):(124.,92.),(0,1):(96.,68.),(0,-1):(96.,120.)}[direction]
    start=SimState("plain",x,y,dashes=1,stamina=92.,frame=137,rem_x=.17,rem_y=-.08)
    target=DashBlockBreakGoal("dash_block_break","plain",x,y,id="dash_break:"+identity,
        identity=identity,rect=(88,80,104,96),clearance_box=(88,80,104,96),
        dash_dx=direction[0],dash_dy=direction[1])
    return sim,start,target


class DashBlockMovesTests(unittest.TestCase):
    def assert_replay(self,sim,start,goal,rows):
        self.assertTrue(rows)
        for actions,final,_ in rows:
            state=start
            for mask in actions:
                state=sim.step(state,mask)
                self.assertFalse(state.dead)
            self.assertEqual(state,final)
            self.assertTrue(goal.achieved(state))
            self.assertIn(goal.identity,state.broken)
            self.assertLess(state.dashes,start.dashes)
            self.assertFalse(start.broken)

    def test_actual_horizontal_and_vertical_collisions_break_only_named_block(self):
        for direction in ((1,0),(-1,0),(0,1),(0,-1)):
            with self.subTest(direction=direction):
                sim,start,goal=fixture(direction)
                rows=dash_block_edges(sim,start,goal,deadline=float("inf"),max_frames=80)
                self.assert_replay(sim,start,goal,rows)
                self.assertEqual(rows[0][1].broken,frozenset({"plain:7"}))
                if direction[0]:
                    self.assertEqual(rows[0][1].vx,-direction[0]*120.)
                    self.assertLess(rows[0][1].vy,0.)

    def test_wrong_identity_indestructible_block_and_no_dash_are_rejected(self):
        for value in (False,"false",0):
            sim,start,goal=fixture(can_dash=value)
            stats={}
            self.assertEqual(dash_block_edges(sim,start,goal,deadline=float("inf"),stats=stats),[])
            self.assertEqual(stats.get("expansions",0),0)
        sim,start,goal=fixture(identity="plain:another")
        self.assertEqual(dash_block_edges(sim,start,goal,deadline=float("inf")),[])
        sim,start,goal=fixture()
        route=dict(waypoints=[dict(room="plain",x=176,y=136,tile_size=8)],
                   segments=[dict(no_dash=True,no_ground=False)])
        for wrapper in (_ConstrainedSimulator(sim,route),
                        _ConstrainedSimulator(sim,{**route,"segments":[{}]},no_dash=True)):
            stats={}
            self.assertEqual(dash_block_edges(wrapper,start,goal,deadline=float("inf"),stats=stats),[])
            self.assertEqual(stats.get("simulated_frames",0),0)
        already=replace(start,broken=frozenset({goal.identity}))
        rows=dash_block_edges(wrapper,already,goal,deadline=float("inf"))
        self.assertEqual(rows,[([],already,"dash_block:already_broken")])

    def test_explicit_public_hit_before_collision_is_kept_without_faking_a_break(self):
        sim,start,goal=fixture()
        public=Goal("target","plain",84,92)
        stage=_StageGoal(goal.kind,goal.room,goal.x,goal.y,id=goal.id,
                         portal=goal,public_goal=public)
        self.assertFalse(public.achieved(start))
        rows=dash_block_edges(sim,start,stage,deadline=float("inf"),max_edges=4)
        before_break=[row for row in rows if not row[1].broken]
        self.assertTrue(before_break)
        for actions,final,label in before_break:
            state=start
            for mask in actions:
                state=sim.step(state,mask)
            self.assertEqual(state,final)
            self.assertTrue(public.achieved(state))
            self.assertFalse(goal.achieved(state))
            self.assertIn("public_hit",label)

    def test_no_ground_constraint_rejects_a_recipe_that_lands_before_breaking(self):
        sim,start,goal=fixture()
        room=sim.map.rooms["plain"]
        room=replace(room,entities=(replace(room.entities[0],y=144,height=32),))
        sim=Simulator(MapData("landing-dash-block",{"plain":room},Path("landing-dash-block.bin")))
        goal=replace(goal,rect=(88,144,104,176))
        route=dict(waypoints=[dict(room="plain",x=176,y=136,tile_size=8)],
                   segments=[dict(no_dash=False,no_ground=True)])
        # A real airborne fall must land before this forced delayed recipe;
        # even a later valid dash collision cannot erase that earlier violation.
        start=replace(start,y=172.,vy=100.)
        wrapped=_ConstrainedSimulator(sim,route,phase=RouteProgress(left_ground=True))
        recipe=((0,)*30+(RIGHT|DASH,)*20,"delayed_after_landing")
        with patch("celeste_search.dash_block_moves._recipes",return_value=iter([recipe])):
            allowed=dash_block_edges(sim,start,goal,deadline=float("inf"),max_frames=80)
        self.assert_replay(sim,start,goal,allowed)
        with patch("celeste_search.dash_block_moves._recipes",return_value=iter([recipe])):
            rows=dash_block_edges(wrapped,start,goal,deadline=float("inf"),max_frames=80)
        self.assertEqual(rows,[])

    def test_slices_resume_recipe_state_without_replaying_or_changing_results(self):
        sim,start,goal=fixture()
        start=replace(start,dash_cooldown=18,prev_input=DASH)
        expected_stats={}
        expected=dash_block_edges(sim,start,goal,deadline=float("inf"),max_frames=80,
            stats=expected_stats,max_expansions=28,remember_results=False)
        clock=[0.];raw=sim.step;steps=[]
        def timed(state,mask):
            clock[0]+=.001
            steps.append((state,mask))
            return raw(state,mask)
        stats={};handle=MacroContinuation()
        with patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]), \
             patch.object(sim,"step",side_effect=timed):
            for _ in range(200):
                actual=dash_block_edges(sim,start,goal,deadline=clock[0]+.006,max_frames=80,
                    stats=stats,max_expansions=28,continuation=handle,remember_results=False)
                if handle.done:
                    break
        self.assertTrue(handle.done)
        self.assertGreater(handle.resumes,10)
        self.assertEqual(actual,expected)
        self.assertEqual(stats,expected_stats)
        self.assertEqual(len(steps),stats["simulated_frames"])
        self.assert_replay(sim,start,goal,actual)

    def test_continuation_rejects_a_different_full_origin_and_releases_state(self):
        sim,start,goal=fixture()
        handle=MacroContinuation();stats={}
        dash_block_edges(sim,start,goal,deadline=0.,stats=stats,continuation=handle)
        with self.assertRaises(ValueError):
            dash_block_edges(sim,replace(start,rem_x=.18),goal,deadline=float("inf"),
                             stats=stats,continuation=handle)
        handle.release()
        self.assertTrue(handle.done)
        self.assertIsNone(handle.work)
        self.assertEqual(handle.results,[])


if __name__=="__main__":
    unittest.main()
