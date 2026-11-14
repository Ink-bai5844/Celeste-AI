"""Region selection reaches the public goal through real, continuous inputs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from celeste_search.assist import SharedAssistState
from celeste_search.assist_gui import AssistWindow, RouteDraft
from celeste_search.assist_planning import _selected_goal, plan_to_route, target_reached
from celeste_search.assist_routes import (normalize_route, start_route_progress,
    advance_route_progress, route_complete)
from celeste_search.assist_targets import expand_exit_target, room_exit_targets
from celeste_search.maps import MapData, Room
from celeste_search.physics import SimState, Simulator, RIGHT
from celeste_search.planner import Heuristic


def fixture():
    rooms={name:Room(name,index*64,0,64,64,("0"*8,)*7+("1"*8,),
                     (),((index*64+16.,56.),)) for index,name in enumerate("ab")}
    data=MapData("region-integration",rooms,Path("region-integration.bin"))
    sim=Simulator(data)
    selected=next(point for point in room_exit_targets(data,"a")
                  if point["edge"]=="right" and point["y"]==8)
    return data,sim,sim.initial_state("a"),selected


class RegionIntegrationTests(unittest.TestCase):
    def test_goal_and_route_accept_another_lane_only_with_explicit_region(self):
        data,sim,start,point=fixture()
        region=expand_exit_target(data,point)
        goal,normalized=_selected_goal(region)
        self.assertEqual(normalized,region)
        self.assertTrue(goal.exit_region)
        route=normalize_route(data,{"waypoints":[region]})
        phase=start_route_progress(route,start)
        state=start
        for _ in range(80):
            before=state
            state=sim.step(state,RIGHT)
            phase=advance_route_progress(route,phase,before,state,RIGHT)
            if state.room!=start.room:
                break
        self.assertFalse(state.dead)
        self.assertTrue(goal.achieved(state))
        self.assertTrue(route_complete(route,phase))
        self.assertFalse(target_reached(state,point))
        self.assertEqual(start_route_progress(route,state).index,0)
        self.assertFalse(replace(goal,crossing_only=True).achieved(replace(state,reason="")))

    def test_source_guidance_uses_nearby_member_instead_of_clicked_lane(self):
        data,sim,start,point=fixture()
        goal,_=_selected_goal(expand_exit_target(data,point))
        guided=replace(goal,room=goal.source_room,x=point["x"]+4,y=point["y"]+9.5)
        x,y=Heuristic(data,guided).waypoint(start)
        self.assertGreater(y,point["y"]+24)
        self.assertAlmostEqual(x,68)
        self.assertLess(abs(y-start.y),8)

    def test_whole_route_replays_and_keeps_original_live_anchor(self):
        data,sim,start,point=fixture()
        route={"waypoints":[expand_exit_target(data,point)]}
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)):
            result=plan_to_route("region-integration.bin",start.to_dict(),route,
                strategy="direct_greedy",seconds=2,expansions=500,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(SimState.from_dict(result["anchor_state"]),start)
        state=start
        for mask in result["actions"]:
            state=Simulator(data).step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))
        self.assertTrue(target_reached(state,result["route"]["waypoints"][0]))
        self.assertFalse(target_reached(state,point))

    def test_named_regional_recipe_keeps_private_landing_and_exact_suffix(self):
        rows=[["0"]*40 for _ in range(20)]
        for y in range(16,20):
            for x in range(8):
                rows[y][x]="1"
        for y in range(12,20):
            for x in range(18,38):
                rows[y][x]="1"
        room=Room("a",0,0,320,160,tuple("".join(row) for row in rows),(),((24.,128.),))
        data=MapData("regional-production",{"a":room},Path("regional-production.bin"))
        simulator=Simulator(data)
        start=simulator.initial_state("a")
        route={"waypoints":[dict(room="a",x=272,y=88)]}
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            result=plan_to_route(str(data.source),start.to_dict(),route,
                strategy="ordered_regional",seconds=8,expansions=3000,max_attempts=1)
        self.assertTrue(result["can_execute"],result)
        self.assertTrue({"ground_region_landing", "ground_frontier_landing"}
                        .intersection(result["strategy_details"]["stages"]))
        self.assertTrue({"ground_exact_suffix", "ground_start_walk"}
                        .intersection(result["strategy_details"]["stages"]))
        self.assertEqual(SimState.from_dict(result["anchor_state"]),start)
        state=start
        raw=Simulator(data)
        for mask in result["actions"]:
            state=raw.step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))
        self.assertTrue(target_reached(state,route["waypoints"][0]))

    def test_new_gui_selection_expands_exit_without_editing_earlier_drafts(self):
        data,_,_,point=fixture()
        window=object.__new__(AssistWindow)
        window.map,window.room=data,data.rooms["a"]
        window.scale,window.offset=1,(0,0)
        window.canvas=SimpleNamespace(canvasx=lambda x:x,canvasy=lambda y:y)
        window.shared=SharedAssistState()
        window.selection=SimpleNamespace(set=Mock())
        window.draft=RouteDraft()
        window.draft.append(point)
        earlier=window.draft.payload()
        window.refresh_editor=Mock()
        window.draw_dynamic=Mock()
        window.select_tile(SimpleNamespace(x=point["x"]+4,y=point["y"]+4))
        self.assertEqual(window.draft.waypoints[0],point)
        self.assertNotIn("exit_region",earlier["waypoints"][0])
        self.assertGreater(len(window.draft.waypoints[1]["exit_region"]),1)
        self.assertTrue(window.shared.commands.empty())


if __name__=="__main__":
    unittest.main()
