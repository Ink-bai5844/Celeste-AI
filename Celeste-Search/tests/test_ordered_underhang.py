"""Underhang detours are geometry proposals in one continuous ordered route."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal, plan_to_route
from celeste_search.assist_strategies import route_strategy_specs
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.underhang_moves import underhang_detour_edges, underhang_edges


def fixture(mirrored=False):
    tiles={(10,y) for y in range(8,14)} | {(14,y) for y in range(8,22)} | {(13,8)}
    entities=[Entity("jumpThru",1,88,64,16,0),Entity("jumpThru",2,88,104,8,0),
              Entity("jumpThru",3,72,88,8,0),Entity("spikesLeft",4,112,72,0,112),
              Entity("strawberry",5,96,80,0,0)]
    if mirrored:
        tiles={(19-x,y) for x,y in tiles}
        entities=[replace(e,x=160-e.x-e.width,
                          kind="spikesRight" if e.kind=="spikesLeft" else e.kind) for e in entities]
    rows=tuple("".join("1" if (x,y) in tiles else "0" for x in range(20)) for y in range(25))
    room=Room("a",0,0,160,200,rows,tuple(entities),((48. if mirrored else 112.,64.),))
    data=MapData("enclosed-underhang",{"a":room},Path("enclosed-underhang.bin"))
    sim=Simulator(data)
    start=replace(sim.initial_state(),frame=123)
    route={"waypoints":[{"room":"a","x":64 if mirrored else 88,"y":72},
                         {"room":"a","x":64 if mirrored else 88,"y":56}]}
    return data,sim,start,route


class OrderedUnderhangTests(unittest.TestCase):
    def test_both_sides_clear_the_adjoining_wall_and_recover_actual_resources(self):
        for mirrored in (False,True):
            with self.subTest(mirrored=mirrored):
                _,sim,start,route=fixture(mirrored)
                goal,_=_selected_goal(route["waypoints"][0])
                self.assertEqual(list(underhang_edges(sim,start,goal,deadline=time.monotonic()+1)),[])
                stats={"max_expansions":40}
                edges=list(underhang_detour_edges(sim,start,goal,deadline=time.monotonic()+2,stats=stats))
                self.assertTrue(edges)
                for actions,end,label in edges:
                    trace=sim.trace(start,actions)
                    self.assertEqual(trace[-1],end)
                    self.assertTrue(goal.achieved(end))
                    self.assertEqual(end.dashes,1)
                    self.assertTrue(all(not state.dead and state.room==start.room for state in trace))
                    # Dash is spent only after moving below the wall, then a
                    # real lower landing refills it before the first waypoint.
                    dash=next(i for i,mask in enumerate(actions) if mask&DASH)
                    self.assertGreater(trace[dash].y,112)
                    self.assertTrue(any(state.on_ground and state.y==104 for state in trace[dash+1:]))
                self.assertGreaterEqual(stats["simulated_frames"],sum(len(edge[0]) for edge in edges))

    def test_ordered_route_replays_both_waypoints_with_one_late_dash_and_safe_tail(self):
        for mirrored in (False,True):
            with self.subTest(mirrored=mirrored):
                data,sim,start,route=fixture(mirrored)
                with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                     patch("celeste_search.assist_planning.macros",return_value=[]), \
                     patch.object(sim,"initial_state",side_effect=AssertionError("no segment reset")):
                    result=plan_to_route(str(data.source),start.to_dict(),route,
                                         strategy="ordered_semantic",seconds=2,expansions=300,max_attempts=1)
                self.assertTrue(result["can_execute"],result)
                trace=sim.trace(start,result["actions"])
                self.assertEqual(trace[-1],SimState.from_dict(result["final_state"]))
                self.assertEqual(len(result["waypoint_indices"]),2)
                self.assertEqual(result["waypoint_indices"][-1],len(result["actions"]))
                self.assertEqual(len(result["dash_start_frames"]),1)
                self.assertLess(result["dash_start_frames"][0],result["waypoint_indices"][0])
                tail=sim.trace(trace[-1],[0]*120)
                self.assertTrue(all(not state.dead for state in tail))
                self.assertEqual(tail[-1].y,64)
                self.assertTrue(tail[-1].on_ground)
                self.assertIn("a:5",tail[-1].berries)

    def test_partial_prefix_is_never_executable_and_constraints_disable_detours(self):
        data,sim,start,route=fixture()
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch("celeste_search.assist_planning.macros",return_value=[]), \
             patch("celeste_search.assist_planning.underhang_climb_macros",return_value=[]):
            result=plan_to_route(str(data.source),start.to_dict(),route,
                                 strategy="ordered_semantic",seconds=1,expansions=100,max_attempts=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertEqual(result["best_prediction"]["waypoints_reached"],1)
        for flag in ("no_dash","no_ground"):
            restricted={**route,"segments":[{},{flag:True}]}
            with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                 patch("celeste_search.assist_planning.macros",return_value=[]), \
                 patch("celeste_search.assist_planning.underhang_detour_edges") as detour:
                result=plan_to_route(str(data.source),start.to_dict(),restricted,
                                     strategy="ordered_semantic",seconds=1,expansions=10,max_attempts=1)
            detour.assert_not_called()
            self.assertFalse(result["can_execute"])

    def test_portfolio_priority_is_scoped_to_unrestricted_underhang_geometry(self):
        data,_,start,route=fixture()
        self.assertEqual(route_strategy_specs(route,start,data)[0]["name"],"ordered_semantic")
        for other in ({**route,"segments":[{"no_dash":True},{}]},
                      {"waypoints":[{"room":"a","x":88,"y":32},{"room":"a","x":96,"y":32}]}):
            self.assertEqual(route_strategy_specs(other,start,data)[0]["name"],"ordered_corridor")

    def test_deadlines_and_budget_cannot_emit_partial_detours(self):
        _,sim,start,route=fixture();goal,_=_selected_goal(route["waypoints"][0])
        for options in ({"deadline":time.monotonic()-1},
                        {"deadline":time.monotonic()+1,"max_frames":5},
                        {"deadline":time.monotonic()+1,"stats":{"max_expansions":0}}):
            self.assertEqual(list(underhang_detour_edges(sim,start,goal,**options)),[])


if __name__=="__main__":
    unittest.main()
