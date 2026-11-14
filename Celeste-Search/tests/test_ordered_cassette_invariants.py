"""Cassette support guidance remains an edge in one complete ordered search."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route, search_support_graph
from celeste_search.assist_strategies import route_strategy_specs
from celeste_search.physics import SimState
from celeste_search.planner import SearchResult
from test_cassette_physics import fixture


ROUTE={"waypoints":[{"room":"a","x":72,"y":56,"tile_size":8},
                    {"room":"a","x":104,"y":56,"tile_size":8}]}


class OrderedCassetteInvariantTests(unittest.TestCase):
    def test_physical_support_edges_continue_exact_waypoint_state_and_phase(self):
        data,sim,start=fixture()
        start=replace(start,frame=123)
        anchors=[]
        def audited(simulator,anchor,goal,**options):
            result=search_support_graph(simulator,anchor,goal,**options)
            anchors.append((anchor,result))
            return result
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch("celeste_search.assist_planning.search_support_graph",side_effect=audited), \
             patch("celeste_search.assist_planning.macros",return_value=[]), \
             patch.object(sim,"initial_state",side_effect=AssertionError("must not reset")):
            result=plan_to_route(str(data.source),start.to_dict(),ROUTE,
                                 strategy="ordered_semantic",seconds=2,
                                 max_attempts=1,expansions=300)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(len(anchors),2)
        first_anchor,first=anchors[0]
        self.assertEqual(anchors[1][0],sim.trace(first_anchor,first.actions)[-1])
        self.assertNotEqual(first_anchor.cassette_clock,anchors[1][0].cassette_clock)
        trace=sim.trace(start,result["actions"])
        self.assertEqual(trace[-1],SimState.from_dict(result["final_state"]))
        self.assertTrue(all(not state.dead for state in trace))
        self.assertEqual(result["waypoint_indices"][-1],len(result["actions"]))
        self.assertEqual([stage["phase"] for stage in result["strategy_details"]["support_stages"]],[0,1])

    def test_any_segment_constraint_disables_unrestricted_support_staging(self):
        for flag in ("no_dash","no_ground"):
            data,sim,start=fixture()
            route={**ROUTE,"segments":[{},{flag:True}]}
            with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                 patch("celeste_search.assist_planning.macros",return_value=[]), \
                 patch("celeste_search.assist_planning.search_support_graph") as graph:
                result=plan_to_route(str(data.source),start.to_dict(),route,
                                     strategy="ordered_semantic",seconds=1,
                                     max_attempts=1,expansions=5)
            graph.assert_not_called()
            self.assertFalse(result["can_execute"])
            self.assertEqual(result["actions"],[])

    def test_successful_first_leg_is_not_published_when_suffix_fails(self):
        data,sim,start=fixture()
        def prefix_only(simulator,anchor,goal,**options):
            if goal.tile_x==72:
                return search_support_graph(simulator,anchor,goal,**options)
            return SearchResult([],anchor,False,1,0,"fixture_suffix_unreachable")
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch("celeste_search.assist_planning.search_support_graph",side_effect=prefix_only), \
             patch("celeste_search.assist_planning.macros",return_value=[]):
            result=plan_to_route(str(data.source),start.to_dict(),ROUTE,
                                 strategy="ordered_semantic",seconds=2,
                                 max_attempts=1,expansions=100)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertEqual(SimState.from_dict(result["final_state"]),start)
        self.assertEqual(result["best_prediction"]["waypoints_reached"],1)

    def test_semantic_priority_and_budget_are_scoped_to_unconstrained_cassette_routes(self):
        data,_,state=fixture()
        selected=route_strategy_specs(ROUTE,state,data)
        self.assertEqual(len(selected),14)
        self.assertEqual(selected[0]["name"],"ordered_semantic")
        self.assertEqual(selected[0]["budget_scale"],2.5)
        for route,anchor in ((ROUTE,replace(state,cassette_clock=None)),
                             ({**ROUTE,"segments":[{},{"no_dash":True}]},state)):
            specs=route_strategy_specs(route,anchor,data)
            self.assertEqual(specs[0]["name"],"ordered_corridor")
            self.assertFalse(any("budget_scale" in spec for spec in specs
                                 if spec["name"] not in {"ordered_corridor","ordered_regional"}))


if __name__=="__main__":
    unittest.main()
