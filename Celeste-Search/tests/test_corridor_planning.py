"""A geometric chain is guidance; only continuous public-route replay succeeds."""
from dataclasses import replace
from types import SimpleNamespace
import json
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import route_complete
from celeste_search.corridor_planning import search_corridor_routes, _ConstrainedSimulator
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import LEFT, RIGHT, JUMP, SimState
from celeste_search.planner import Goal, SearchResult


class BranchSimulator:
    map_data = None
    last_step_started_dash = False

    def step(self, state, mask):
        if mask == 0:
            return replace(state, frame=state.frame+1, prev_input=mask)
        if state.x == 0 and mask in (LEFT, RIGHT):
            return replace(state, x=20., vx=-1. if mask == LEFT else 1.,
                           frame=state.frame+1, prev_input=mask)
        if state.x == 20 and state.vx > 0 and mask == RIGHT:
            return replace(state, x=84., frame=state.frame+1, prev_input=mask)
        return replace(state, dead=True, reason="no_edge", frame=state.frame+1)


def corridor(identity, portal_x):
    portal = Goal("corridor", "a", portal_x, 8., id=f"portal:{portal_x}")
    return CorridorRoute(identity,"a",(portal,),((0.,8.),(portal_x,8.),(84.,8.)),(),100.)


class CorridorPlanningTests(unittest.TestCase):
    def setUp(self):
        self.sim = BranchSimulator()
        self.start = SimState("a",0.,8.,on_ground=False)
        self.route = dict(waypoints=[dict(room="a",x=80,y=0,tile_size=8)],
                          segments=[dict(no_dash=False,no_ground=False)])
        self.goal = _selected_goal(self.route["waypoints"][0])[0]

    def solver(self, corridors, policy):
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=corridors), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=policy):
            return search_corridor_routes(self.sim,self.start,self.route,[self.goal],
                deadline=time.monotonic()+2,max_expansions=100,max_frames=20)

    def policy(self, sim, start, goal, **kwargs):
        actions=[]
        if goal.x == 20 and start.x == 0:
            actions=[LEFT if sim.no_dash else RIGHT]
        elif goal.x == self.goal.x and start.x == 20 and start.vx > 0:
            actions=[RIGHT]
        current=start
        for mask in actions:
            current=sim.step(current,mask)
        return SearchResult(actions,current,bool(actions and goal.achieved(current)),1,len(actions),"fixture")

    def test_failed_corridor_then_next_and_suffix_arrival_backtracking(self):
        result=self.solver([corridor("blocked",40),corridor("open",20)],self.policy)
        actions,final,reached,_,_,phase,details=result
        self.assertTrue(reached)
        self.assertEqual(actions,[RIGHT,RIGHT])
        self.assertEqual(final.x,84.)
        self.assertTrue(route_complete(self.route,phase))
        inventory=details["topology"]["corridors"]
        self.assertEqual([row["status"] for row in inventory],["pending","trying"])
        self.assertEqual(details["selected_corridor_id"],"open")
        self.assertGreaterEqual(details["backtracks"],1)
        self.assertTrue(any(row["reason"]=="suffix_failed_backtrack"
                            for row in details["corridor_attempts"]))
        json.dumps(details)  # Public metadata never contains Goal/SimState objects.

    def test_private_portal_does_not_complete_public_target(self):
        def prefix_only(sim,start,goal,**kwargs):
            if goal.kind=="target":
                return SearchResult([],start,False,1,0,"budget")
            return self.policy(sim,start,goal,**kwargs)
        result=self.solver([corridor("prefix",20)],prefix_only)
        self.assertFalse(result[2])
        self.assertFalse(route_complete(self.route,result[5]))
        self.assertEqual(result[5].index,0)
        self.assertIsNone(result[6]["selected_corridor_id"])
        self.assertEqual(result[6]["topology"]["corridors"][0]["status"],"pending")
        self.assertIn("预算",result[6]["topology"]["corridors"][0]["reason"])

    def test_budget_limited_shared_portal_resumes_across_geometric_branches(self):
        calls=0;handles=[]
        def later_available(sim,start,goal,**kwargs):
            nonlocal calls
            if goal.x==20 and start.x==0:
                calls+=1
                handles.append(((sim.no_dash,kwargs["variant"]),kwargs["continuation"]))
                if calls<=3:
                    return SearchResult([],start,False,1,0,"budget")
            return self.policy(sim,start,goal,**kwargs)
        result=self.solver([corridor("first",20),corridor("second",20)],later_available)
        self.assertTrue(result[2])
        self.assertIn(result[6]["selected_corridor_id"],{"first","second"})
        self.assertGreater(calls,3)
        for recipe in {key for key,_ in handles}:
            self.assertEqual(len({id(handle) for key,handle in handles if key==recipe}),1)

    def test_same_arrival_with_shorter_prefix_can_retry_a_frame_limited_suffix(self):
        class Sim(BranchSimulator):
            def step(self,state,mask):
                x=({(0,LEFT):10.,(0,RIGHT):20.,(10,RIGHT):20.,
                     (20,RIGHT):40.,(40,RIGHT):84.}).get((state.x,mask))
                return replace(state,x=state.x if x is None else x,dead=x is None,
                               frame=state.frame+1,prev_input=mask)
        self.sim=Sim()
        def policy(sim,start,goal,**kwargs):
            actions=[]
            if start.x==0 and goal.kind!="target":
                actions=[LEFT,RIGHT] if sim.no_dash else [RIGHT]
            elif start.x==20 and goal.kind=="target" and kwargs["max_frames"]>=2:
                actions=[RIGHT,RIGHT]
            state=start
            for mask in actions:state=sim.step(state,mask)
            return SearchResult(actions,state,bool(actions and goal.achieved(state)),1,len(actions),"fixture")
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor("short",20)]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=policy):
            result=search_corridor_routes(self.sim,self.start,self.route,[self.goal],
                deadline=time.monotonic()+2,max_expansions=100,max_frames=3)
        self.assertTrue(result[2])
        self.assertEqual(result[0],[RIGHT,RIGHT,RIGHT])
        self.assertGreater(result[6]["backtracks"],0)

    def test_no_ground_allows_takeoff_but_rejects_landing_and_upward_probe(self):
        route={**self.route,"segments":[dict(no_dash=False,no_ground=True)]}
        class Sim:
            map_data=None
            def __init__(self, after):self.after=after
            def step(self,before,mask):return self.after
        grounded=replace(self.start,on_ground=True,vy=0)
        airborne=replace(self.start,on_ground=False,vy=-20)
        up_probe=replace(self.start,on_ground=True,vy=-20)
        for before,after,dead in ((grounded,airborne,False),(airborne,grounded,True),
                                  (up_probe,grounded,True),(grounded,up_probe,False)):
            value=_ConstrainedSimulator(Sim(after),route).step(before,JUMP)
            self.assertEqual(value.dead,dead)

    def test_mismatched_public_goal_count_is_rejected_before_geometry(self):
        self.route["waypoints"]*=2
        self.route["segments"]*=2
        with patch("celeste_search.corridor_planning.generate_corridor_routes") as generate:
            value=search_corridor_routes(self.sim,self.start,self.route,[self.goal],
                deadline=time.monotonic()+2,max_expansions=100,max_frames=20)
        self.assertFalse(value[2]);self.assertEqual(value[0],[])
        self.assertEqual(value[6]["reason"],"unsupported_guidance_combination")
        generate.assert_not_called()

    def test_slow_local_probe_preserves_cumulative_depth_across_resumed_quanta(self):
        clock=[0.]
        calls=[]
        spent=0
        handle_ids=[]
        wrapper_ids=[]
        def search(sim,start,goal,**kwargs):
            nonlocal spent
            allowance=kwargs["max_expansions"]
            calls.append((sim.no_dash,allowance,kwargs["deadline"]))
            self.assertLessEqual(allowance,64)
            if start.x==0 and goal.x!=20:
                return SearchResult([],start,False,1,0,"fixture")
            if start.x==0 and sim.no_dash:
                clock[0]+=.7 if len(calls)==1 else .01
                return SearchResult([],start,False,allowance,0,"budget")
            if start.x==0:
                if kwargs["variant"]!=4:
                    return SearchResult([],start,False,1,0,"fixture")
                spent+=allowance
                handle_ids.append(id(kwargs["continuation"]))
                wrapper_ids.append(id(sim))
                clock[0]+=.01
                if spent<817:
                    return SearchResult([],start,False,allowance,0,"budget")
                final=sim.step(start,RIGHT)
                return SearchResult([RIGHT],final,True,allowance,1,"fixture")
            final=sim.step(start,RIGHT)
            return SearchResult([RIGHT],final,goal.achieved(final),1,1,"fixture")
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor("slow",20)]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.corridor_planning.time.monotonic",side_effect=lambda:clock[0]):
            result=search_corridor_routes(self.sim,self.start,self.route,[self.goal],
                deadline=2.,max_expansions=3000,max_frames=20)
        self.assertTrue(result[2])
        self.assertEqual(result[0],[RIGHT,RIGHT])
        self.assertTrue(all(deadline<=2. for _,_,deadline in calls))
        self.assertEqual(calls[0][1],4)
        self.assertGreaterEqual(spent,817)
        self.assertEqual(len(set(handle_ids)),1)
        self.assertEqual(len(set(wrapper_ids)),1)


if __name__=="__main__":unittest.main()
