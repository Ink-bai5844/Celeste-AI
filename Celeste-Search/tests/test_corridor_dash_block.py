"""Conditional dash-block edges must break, then physically pass the opening."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import (CorridorRoute,DashBlockApproachGoal,
                                           DashBlockPassageGoal)
from celeste_search.physics import RIGHT
from celeste_search.planner import SearchResult
from test_dash_block_moves import fixture


class CorridorDashBlockTests(unittest.TestCase):
    def run_case(self, *, no_dash=False, blocked=False):
        sim,start,broken=fixture(can_dash=not blocked)
        route=normalize_route(sim.map,dict(room="plain",x=176,y=168,no_dash=no_dash))
        # Segment limits are explicit; a target dictionary is not a substitute
        # for the ordered route's original per-segment restriction.
        route["segments"][0]["no_dash"]=no_dash
        goal=_selected_goal(route["waypoints"][0])[0]
        approach=DashBlockApproachGoal("dash_approach","plain",start.x,start.y,
            id="approach",identity=broken.identity,bounds=(start.x-2,start.y-2,start.x+2,start.y+2))
        passage=DashBlockPassageGoal("corridor","plain",96,92,id="passage",
            identity=broken.identity,bounds=(88,80,104,107))
        corridor=CorridorRoute("destructible","plain",(approach,broken,passage),
            ((start.x,start.y),(96.,92.),(180.,176.)),(),180.,(broken.identity,))
        calls=[]
        def ordinary(wrapped,origin,target,**kwargs):
            calls.append((target.kind,origin,wrapped.no_dash))
            if target.kind=="dash_block_break":
                # Exercise the new real collision family, not a mocked break.
                return SearchResult([],origin,False,1,0,"frontier_exhausted")
            state=origin;actions=[]
            for _ in range(min(160,kwargs["max_frames"])):
                if target.achieved(state) or state.dead:
                    break
                state=wrapped.step(state,RIGHT)
                actions.append(RIGHT)
            return SearchResult(actions,state,target.achieved(state),1,len(actions),"frontier_exhausted")
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=ordinary):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+3.,
                max_expansions=200,max_frames=300)
        return sim,start,route,result,calls

    def test_actual_break_rebound_and_passage_form_one_continuous_public_route(self):
        sim,start,route,result,calls=self.run_case()
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertIn("plain:7",result[1].broken)
        passage_starts=[state for kind,state,_ in calls if kind=="corridor"]
        self.assertTrue(passage_starts)
        self.assertTrue(all("plain:7" in state.broken for state in passage_starts))
        self.assertTrue(any(state.vx<0 and state.vy<0 for state in passage_starts),
                        "The suffix must inherit the actual collision rebound")
        state=start
        for mask in result[0]:
            state=sim.step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,result[1])

    def test_forbidden_or_impossible_break_cannot_skip_to_passage(self):
        for kwargs in (dict(no_dash=True),dict(blocked=True)):
            with self.subTest(kwargs=kwargs):
                _,_,route,result,calls=self.run_case(**kwargs)
                self.assertFalse(result[2])
                self.assertFalse(route_complete(route,result[5]))
                self.assertFalse(any(kind=="corridor" for kind,_,_ in calls))

    def test_required_break_does_not_spend_search_quanta_on_impossible_no_dash_probe(self):
        _,_,_,result,calls=self.run_case()
        self.assertTrue(result[2],result[6])
        self.assertFalse(any(kind=="dash_block_break" and no_dash
                             for kind,_,no_dash in calls),
                         "An unbroken block without a live attack requires a real dash")


if __name__=="__main__":
    unittest.main()
