"""A user route is one continuous life with constraints at exact hit frames."""
from dataclasses import replace
from pathlib import Path
import unittest
import time
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route, target_reached, RestingSupportGoal
from celeste_search.assist_routes import (normalize_route, start_route_progress,
                                         advance_route_progress, route_complete)
from celeste_search.assist_validation import AssistValidationError
from celeste_search.assist_targets import room_exit_targets
from celeste_search.maps import MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import Goal, SearchResult, search_leg


def fixture():
    rows=("0"*24,)*12+("1"*24,)*2
    rooms={name:Room(name,index*192,0,192,112,rows,(),((index*192+16.,96.),))
           for index,name in enumerate("ab")}
    return MapData("route",rooms,Path("route.bin"))


def point(x,y=88,room="a"):
    return {"room":room,"x":x,"y":y,"tile_size":8}


class RouteConstraintTests(unittest.TestCase):
    def setUp(self):
        self.data=fixture()
        self.start=Simulator(self.data).initial_state("a")

    def test_normalization_accepts_single_target_and_checks_segment_types(self):
        route=normalize_route(self.data,point(40))
        self.assertEqual(route["segments"],[dict(no_dash=False,no_ground=False)])
        for segments in ([],[{"no_dash":1}],[{"no_ground":"true"}],[{"unknown":True}]):
            with self.subTest(segments=segments),self.assertRaises(AssistValidationError):
                normalize_route(self.data,{"waypoints":[point(40)],"segments":segments})

    def test_visiting_later_point_does_not_skip_required_order(self):
        route=normalize_route(self.data,{"waypoints":[point(80),point(40)]})
        progress=start_route_progress(route,self.start)
        later=replace(self.start,x=44)
        progress=advance_route_progress(route,progress,self.start,later,1)
        self.assertEqual(progress.index,0)
        first=replace(later,x=84)
        progress=advance_route_progress(route,progress,later,first,1)
        self.assertEqual(progress.index,1)
        progress=advance_route_progress(route,progress,first,later,2)
        self.assertTrue(route_complete(route,progress))

    def test_no_ground_cannot_be_satisfied_by_walking(self):
        route=normalize_route(self.data,{"waypoints":[point(40)],"segments":[{"no_ground":True}]})
        progress=start_route_progress(route,self.start)
        progress=advance_route_progress(route,progress,self.start,replace(self.start,x=44),1)
        self.assertEqual(progress.index,0)
        self.assertFalse(progress.left_ground)
        self.assertIsNone(progress.violation)

    def test_no_ground_allows_takeoff_but_rejects_landing_at_target(self):
        route=normalize_route(self.data,{"waypoints":[point(40)],"segments":[{"no_ground":True}]})
        progress=start_route_progress(route,self.start)
        air=replace(self.start,x=30,y=80,on_ground=False)
        progress=advance_route_progress(route,progress,self.start,air,17)
        self.assertTrue(progress.left_ground)
        landing=replace(self.start,x=44)
        progress=advance_route_progress(route,progress,air,landing,1)
        self.assertEqual(progress.violation,"landing_forbidden")
        self.assertEqual(progress.index,0)

    def test_no_dash_rejects_input_active_dash_and_pending_buffer(self):
        route=normalize_route(self.data,{"waypoints":[point(40)],"segments":[{"no_dash":True}]})
        progress=start_route_progress(route,self.start)
        self.assertEqual(advance_route_progress(route,progress,self.start,self.start,33).violation,"dash_forbidden")
        for state in (replace(self.start,state=2),replace(self.start,dash_frames=1),replace(self.start,dash_buffer=1)):
            self.assertEqual(start_route_progress(route,state).violation,"dash_forbidden")

    def test_active_dash_cannot_spill_into_next_protected_segment(self):
        route=normalize_route(self.data,{"waypoints":[point(40),point(80)],
                            "segments":[{}, {"no_dash":True}]})
        progress=start_route_progress(route,self.start)
        arrival=replace(self.start,x=44,state=2,dash_frames=8)
        progress=advance_route_progress(route,progress,self.start,arrival,33)
        self.assertEqual(progress.index,1)
        self.assertEqual(progress.violation,"dash_forbidden")

    def test_live_telemetry_and_simulator_use_identical_rules(self):
        route=normalize_route(self.data,{"waypoints":[point(40)],"segments":[{"no_ground":True}]})
        before={"room":"a","x":16,"y":80,"on_ground":False,"dead":False,
                "model_state":{"state":0,"dash_frames":0,"dash_buffer":0}}
        after={**before,"x":44,"y":96,"on_ground":True}
        progress=start_route_progress(route,before)
        self.assertEqual(advance_route_progress(route,progress,before,after,1).violation,"landing_forbidden")

    def test_rising_through_one_way_support_probe_is_not_landing(self):
        route=normalize_route(self.data,{"waypoints":[point(40,64),point(80,64)],
                                       "segments":[{"no_ground":True}]*2})
        before=replace(self.start,x=32,y=76,vy=-90,on_ground=False)
        rising=replace(before,x=44,y=71,vy=-75,on_ground=True)
        for as_telemetry in (False,True):
            with self.subTest(as_telemetry=as_telemetry):
                first=before.to_dict() if as_telemetry else before
                second=rising.to_dict() if as_telemetry else rising
                progress=start_route_progress(route,first)
                progress=advance_route_progress(route,progress,first,second,17)
                self.assertIsNone(progress.violation)
                self.assertEqual(progress.index,1)
                self.assertTrue(progress.left_ground)
                self.assertTrue(start_route_progress(route,second).left_ground)

    def test_descending_support_at_endpoint_still_violates_no_landing(self):
        route=normalize_route(self.data,{"waypoints":[point(40,64)],
                                       "segments":[{"no_ground":True}]})
        before=replace(self.start,x=32,y=66,vy=45,on_ground=False)
        for vertical_speed in (0,45):
            with self.subTest(vertical_speed=vertical_speed):
                landing=replace(before,x=44,y=71,vy=vertical_speed,on_ground=True)
                progress=advance_route_progress(route,start_route_progress(route,before),before,landing,1)
                self.assertEqual(progress.violation,"landing_forbidden")
                self.assertEqual(progress.index,0)

    def test_death_invalidates_whole_route(self):
        route=normalize_route(self.data,{"waypoints":[point(40),point(80)]})
        progress=start_route_progress(route,self.start)
        progress=advance_route_progress(route,progress,self.start,replace(self.start,dead=True),0)
        self.assertEqual(progress.violation,"player_died")
        progress=advance_route_progress(route,progress,self.start,replace(self.start,x=84),0)
        self.assertFalse(route_complete(route,progress))

    def test_exterior_point_requires_directed_crossing_and_cannot_double_count(self):
        target=next(target for target in room_exit_targets(self.data,"a")
                    if target["edge"]=="right" and target["y"]==88)
        route=normalize_route(self.data,{"waypoints":[target,target]})
        neighbor=replace(self.start,room="b",x=196)
        self.assertTrue(target_reached(neighbor,target))
        self.assertEqual(start_route_progress(route,neighbor).index,0)
        progress=start_route_progress(route,self.start)
        progress=advance_route_progress(route,progress,self.start,neighbor,1)
        self.assertEqual(progress.index,1)
        progress=advance_route_progress(route,progress,neighbor,replace(neighbor,x=197),1)
        self.assertEqual(progress.index,1)


class RoutePlanningTests(unittest.TestCase):
    def plan(self,data,sim,start,route,**kwargs):
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)):
            return plan_to_route("route.bin",start.to_dict(),route,seconds=2,expansions=1500,**kwargs)

    def test_entire_route_replays_ordered_without_resets(self):
        data=fixture();sim=Simulator(data)
        start=replace(sim.initial_state("a"),x=24,frame=300,berries=frozenset({"past:1"}))
        route={"waypoints":[point(48),point(112)]}
        result=self.plan(data,sim,start,route)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["waypoint_indices"][-1],len(result["actions"]))
        self.assertEqual(len(result["expected"]),len(result["actions"])+1)
        self.assertEqual(SimState.from_dict(result["anchor_state"]),start)
        state=start
        phase=start_route_progress(result["route"],state)
        for action in result["actions"]:
            after=sim.step(state,action)
            phase=advance_route_progress(result["route"],phase,state,after,action)
            self.assertIsNone(phase.violation)
            state=after
        self.assertTrue(route_complete(result["route"],phase))
        self.assertEqual(state,SimState.from_dict(result["final_state"]))
        self.assertEqual(state.frame,start.frame+len(result["actions"]))

    def test_two_air_points_no_dash_no_ground_preserve_flight(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        route={"waypoints":[point(40,64),point(56,64)],
               "segments":[{"no_dash":True,"no_ground":True}]*2}
        result=self.plan(data,sim,start,route,max_attempts=2)
        self.assertTrue(result["can_execute"],result)
        self.assertFalse(any(action&32 for action in result["actions"]))
        self.assertTrue(all(not state["on_ground"] for state in result["expected"][1:]))
        self.assertGreater(result["waypoint_indices"][1],result["waypoint_indices"][0])

    def test_budget_exhaustion_does_not_publish_reachable_prefix(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)):
            result=plan_to_route("route.bin",start.to_dict(),{"waypoints":[point(24),point(176)]},
                                 seconds=1,expansions=1,max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertEqual(result["final_state"],result["anchor_state"])

    def test_backtracks_to_alternate_arrival_when_first_suffix_is_dead(self):
        data=fixture();start=SimState("a",16,96,on_ground=True)
        class BranchSimulator:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                if state.x==16:
                    return replace(state,x=40,vx=10 if action==1 else 90,frame=state.frame+1)
                if state.x==40:
                    return replace(state,x=80,dead=state.vx<30,frame=state.frame+1)
                return replace(state,dead=True,frame=state.frame+1)
        sim=BranchSimulator()
        def branches(state,*args):
            return [(1,1),(2,1)] if state.x==16 else [(1,1)]
        with patch("celeste_search.assist_planning.macros",side_effect=branches):
            result=self.plan(data,sim,start,{"waypoints":[point(40),point(80)]},max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[2,1])
        self.assertEqual(result["waypoint_indices"],[1,2])

    def test_exterior_point_is_real_neighbor_entry_then_continues(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        exit_target=next(target for target in room_exit_targets(data,"a")
                         if target["edge"]=="right" and target["y"]==88)
        result=self.plan(data,sim,start,{"waypoints":[exit_target,point(224,88,"b")]})
        self.assertTrue(result["can_execute"],result)
        first=result["expected"][result["waypoint_indices"][0]]
        self.assertEqual(first["room"],"b")
        self.assertTrue(target_reached(first,exit_target))
        self.assertTrue(result["allow_room_continuation"])
        self.assertEqual(result["final_state"]["room"],"b")

    def test_mixed_macro_can_hit_waypoint_before_later_dash_becomes_allowed(self):
        data=fixture();start=SimState("a",16,96,on_ground=True)
        class MacroSimulator:
            map=data
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                return replace(state,x=40 if state.x==16 else 80,
                               state=2 if action&32 else 0,frame=state.frame+1)
        def macros_for(state,*args):
            return [((1,33),2)] if state.x==16 else [(33,1)]
        route={"waypoints":[point(40),point(80)],"segments":[{"no_dash":True},{}]}
        with patch("celeste_search.assist_planning.macros",side_effect=macros_for):
            result=self.plan(data,MacroSimulator(),start,route,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[1,33])
        self.assertEqual(result["waypoint_indices"],[1,2])

    def test_single_target_semantic_proposal_still_requires_surviving_full_replay(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        forged=SearchResult([1],replace(start,x=80),True,1,1,"goal")
        with patch("celeste_search.assist_planning._search_single_assist",return_value=forged), \
             patch.object(sim,"step",return_value=replace(start,dead=True)):
            result=self.plan(data,sim,start,{"waypoints":[point(80)]},max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertEqual(result["final_state"],result["anchor_state"])
        self.assertTrue(result["attempts"][0]["replay_rejected"])

    def test_one_target_internal_landing_waits_real_tick_for_dash_refill(self):
        data=fixture();start=SimState("a",16,96,on_ground=True,dashes=1)
        class RefillOnNextGroundUpdate:
            map=data
            max_dashes=1
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                if state.x==16 and action==37:
                    return replace(state,x=40,y=64,dashes=0,on_ground=True,frame=state.frame+1)
                if state.x==40 and action==0:
                    return replace(state,dashes=1,frame=state.frame+1)
                if state.x==40 and action==1 and state.dashes==1:
                    return replace(state,x=180,y=96,frame=state.frame+1)
                return replace(state,dead=True,frame=state.frame+1)
        simulator=RefillOnNextGroundUpdate()
        def first_leg(sim,state,goal,**kwargs):
            return SearchResult([37],sim.step(state,37),True,1,1,"goal")
        def suffix(sim,state,goal,**kwargs):
            result=sim.step(state,1)
            return SearchResult([1],result,not result.dead,1,1,"goal")
        with patch("celeste_search.assist_planning.coverage_goals",
                   return_value=[Goal("surface","a",40,64)]), \
             patch("celeste_search.assist_planning.search_leg",side_effect=first_leg), \
             patch("celeste_search.assist_planning.search_goal",side_effect=suffix):
            result=self.plan(data,simulator,start,{"waypoints":[point(176)]},max_attempts=1,variant=4)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[37,0,1])
        self.assertEqual(result["waypoint_indices"],[3])
        self.assertEqual(result["expected"][1]["dashes"],0)
        self.assertEqual(result["expected"][2]["dashes"],1)
        self.assertEqual(result["final_state"]["frame"],3)

    def test_internal_support_requires_real_braking_not_a_moving_edge_touch(self):
        data=fixture();sim=Simulator(data)
        start=replace(sim.initial_state("a"),vx=60,frame=230)
        goal=RestingSupportGoal("surface","a",48,96)
        edge_touch=replace(start,x=57,vx=-65,vy=15)
        self.assertTrue(Goal("surface","a",48,96).achieved(edge_touch))
        self.assertFalse(goal.achieved(edge_touch))
        result=search_leg(sim,start,goal,deadline=time.monotonic()+2,max_expansions=1000,variant=4)
        self.assertTrue(result.reached,result.reason)
        replay=start
        for action in result.actions:
            replay=sim.step(replay,action)
            self.assertFalse(replay.dead)
        self.assertEqual(replay,result.state)
        self.assertTrue(goal.achieved(replay))
        self.assertEqual(replay.frame,start.frame+len(result.actions))


if __name__=="__main__":
    unittest.main()
