"""Dash preference is soft, history-aware, and subordinate to route restrictions."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import (_search_ordered_route_pass, _selected_goal,
                                          plan_to_route)
from celeste_search.assist_routes import normalize_route
from celeste_search.physics import DASH, Simulator
from test_assist_routes import fixture, point


class OrderedDashPreferenceTests(unittest.TestCase):
    def test_zero_dash_attempt_precedes_relaxation_for_every_ordered_profile(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state("a")
        route={"waypoints":[point(40),point(80)]}
        for name in ("fine","balanced","greedy","long","short_jump","precision",
                     "micro","lookahead","phase_diverse","wide","refill_wait","semantic"):
            with self.subTest(profile=name), \
                 patch("celeste_search.assist_planning._context",return_value=(data,sim)):
                result=plan_to_route(str(data.source),start.to_dict(),route,
                                     strategy="ordered_"+name,seconds=1,max_attempts=1,expansions=500)
            self.assertTrue(result["can_execute"],result)
            self.assertEqual(result["dash_start_frames"],[])
            self.assertEqual(result["strategy_details"]["dash_preference"],"zero_dash_first")

    def test_relaxation_can_backtrack_before_first_waypoint_for_required_early_dash(self):
        data=fixture();start=Simulator(data).initial_state("a")
        class Fork:
            map=data
            last_step_started_dash=False
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                self.last_step_started_dash=False
                if state.x==16:
                    self.last_step_started_dash=bool(action&DASH)
                    return replace(state,x=40,vx=90 if action&DASH else -90,
                                   dashes=0 if action&DASH else 1,frame=state.frame+1)
                if state.x==40:
                    return replace(state,x=80,dead=state.vx<0,frame=state.frame+1)
                return replace(state,dead=True,frame=state.frame+1)
        sim=Fork();route={"waypoints":[point(40),point(80)]}
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch("celeste_search.assist_planning.macros",return_value=[(DASH,1),(1,1)]):
            result=plan_to_route(str(data.source),start.to_dict(),route,
                                 strategy="ordered_fine",seconds=1,max_attempts=1,expansions=50)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["actions"],[DASH,1])
        self.assertEqual(result["dash_start_frames"],[1])
        self.assertEqual(result["strategy_details"]["dash_preference"],"zero_dash_then_relaxed")
        self.assertGreater(result["strategy_details"]["zero_dash_probe_expansions"],0)
        with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
             patch("celeste_search.assist_planning.macros",return_value=[(DASH,1),(1,1)]):
            result=plan_to_route(str(data.source),start.to_dict(),
                                 {**route,"segments":[{"no_dash":True},{}]},
                                 strategy="ordered_fine",seconds=1,max_attempts=1,expansions=50)
        self.assertFalse(result["can_execute"])

    def test_longer_zero_dash_history_replaces_shorter_dash_history_at_same_state(self):
        data=fixture();start=Simulator(data).initial_state("a")
        class Merge:
            map=data
            last_step_started_dash=False
            def capability_report(self,name):
                return {"unsupported":[]}
            def step(self,state,action):
                self.last_step_started_dash=state.x==16 and bool(action&DASH)
                x=(24 if action&DASH else 20) if state.x==16 else {20:24,24:40,40:80}[state.x]
                # A same-tick refill/interaction may leave an identical final
                # state despite consuming a dash; only the event records it.
                return replace(state,x=x,frame=state.frame+1)
        sim=Merge();route=normalize_route(data,{"waypoints":[point(40),point(80)]})
        goals=[_selected_goal(point)[0] for point in route["waypoints"]]
        def actions(state,*args):
            return [(DASH,1),(1,1)] if state.x==16 else [(1,1)]
        with patch("celeste_search.assist_planning.macros",side_effect=actions):
            result=_search_ordered_route_pass(sim,start,route,goals,deadline=time.monotonic()+1,
                    max_expansions=20,max_frames=20,variant=0,forbidden=set(),allow_dash=True)
        self.assertTrue(result[2])
        self.assertEqual(result[0],[1,1,1,1])
        self.assertEqual(result[-1]["actual_dash_count"],0)


if __name__=="__main__":
    unittest.main()
