"""One public-point arrival cannot starve better states on its own corridor."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import advance_route_progress, route_complete, start_route_progress
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import DASH, LEFT, RIGHT, SimState
from celeste_search.planner import Goal, SearchResult


class ArrivalSimulator:
    """Deterministic physical branch fixture; both first arrivals hit x20."""
    map_data = None
    last_step_started_dash = False

    def __init__(self, three_points=False, land_final=False):
        self.three_points = three_points
        self.land_final = land_final

    def step(self, state, mask):
        self.last_step_started_dash = bool(mask & DASH)
        if state.x == 0 and mask in (LEFT, RIGHT):
            return replace(state, x=20., vx=-1. if mask == LEFT else 1.,
                           frame=state.frame+1, prev_input=mask)
        if state.x == 20 and state.vx > 0 and mask & RIGHT:
            x = 60. if self.three_points else 84.
            return replace(state, x=x, on_ground=self.land_final and x==84.,
                           frame=state.frame+1, prev_input=mask)
        if state.x == 60 and mask & RIGHT:
            return replace(state, x=84., on_ground=self.land_final,
                           frame=state.frame+1, prev_input=mask)
        return replace(state, dead=True, reason="no_edge", frame=state.frame+1)


class PublicArrivalFairnessTests(unittest.TestCase):
    def exercise(self, *, three_points=False, no_dash=False, no_ground=False):
        sim = ArrivalSimulator(three_points, land_final=no_ground)
        start = SimState("a", 0., 8., on_ground=False, dashes=1)
        xs = (16, 56, 80) if three_points else (16, 80)
        route = dict(waypoints=[dict(room="a",x=x,y=0,tile_size=8) for x in xs],
                     segments=[dict(no_dash=False,no_ground=False) for _ in xs])
        route["segments"][-1] = dict(no_dash=no_dash, no_ground=no_ground)
        goals = [_selected_goal(point)[0] for point in route["waypoints"]]
        probes = []

        def corridors(simulator, origin, goal, **kwargs):
            return [CorridorRoute("only", "a", (Goal("corridor", "a",goal.x,goal.y,id="only"),),
                                  ((origin.x,origin.y),(goal.x,goal.y)), (), 80.)]

        def search(wrapped, origin, target, **kwargs):
            probes.append((wrapped.phase.index, origin.vx, wrapped.no_dash))
            if origin.x == 0:
                # The conservative first recipe supplies a legal but useless
                # momentum state. The relaxed recipe reaches the same public
                # point with useful momentum, without actually using a dash.
                actions = [LEFT if wrapped.no_dash else RIGHT]
            elif origin.vx > 0:
                actions = [RIGHT | (DASH if no_dash else 0)]
            else:
                # This branch remains alive and budget-limited; it cannot
                # exhaust naturally before consuming the whole test request.
                return SearchResult([],origin,False,min(4,kwargs["max_expansions"]),0,"budget")
            state = origin
            for mask in actions:
                state = wrapped.step(state,mask)
            return SearchResult(actions,state,target.achieved(state),1,len(actions),"fixture")

        with patch("celeste_search.corridor_planning.generate_corridor_routes",side_effect=corridors), \
                patch("celeste_search.corridor_planning.search_leg",side_effect=search):
            result = search_corridor_routes(sim,start,route,goals,deadline=time.monotonic()+2,
                                            max_expansions=120,max_frames=8)
        state, phase = start, start_route_progress(route,start)
        for mask in result[0]:
            before, state = state, sim.step(state,mask)
            phase = advance_route_progress(route,phase,before,state,mask)
            self.assertFalse(state.dead)
            self.assertFalse(phase.violation)
        self.assertEqual(state,result[1])
        self.assertEqual(phase,result[5])
        return result,route,probes

    def test_bad_first_public_arrival_does_not_starve_second_same_corridor_arrival(self):
        result, route, probes = self.exercise()
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertEqual(result[0],[RIGHT,RIGHT])
        self.assertIn((1,-1.,True),probes,"The bad suffix must receive real search turns")
        self.assertIn((0,0.,False),probes,"Its parent must resume the better arrival recipe")
        first_good = next(i for i,row in enumerate(probes) if row[0]==1 and row[1]>0)
        self.assertGreater(sum(row[0]==1 and row[1]<0 for row in probes[:first_good]),1)
        self.assertLess(result[3],120)

    def test_later_public_stages_keep_exact_prefix_and_order(self):
        result, route, probes = self.exercise(three_points=True)
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertEqual(result[0],[RIGHT,RIGHT,RIGHT])
        self.assertTrue(any(row[0]==2 for row in probes))

    def test_fairness_does_not_relax_original_dash_or_landing_constraints(self):
        for constraints in (dict(no_dash=True),dict(no_ground=True)):
            with self.subTest(constraints=constraints):
                result, route, probes = self.exercise(**constraints)
                self.assertFalse(result[2])
                self.assertFalse(route_complete(route,result[5]))
                self.assertTrue(any(row[0]==1 and row[1]>0 for row in probes),
                                "The second arrival was tested, then its illegal suffix rejected")


if __name__ == "__main__":
    unittest.main()
