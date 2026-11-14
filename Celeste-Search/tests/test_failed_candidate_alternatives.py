"""A failed complete input sequence must not end search at its first rediscovery."""
from contextlib import ExitStack
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import advance_route_progress, route_complete, start_route_progress
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.mechanism_planning import search_mechanism_routes
from celeste_search.physics import LEFT, RIGHT, SimState
from celeste_search.planner import SearchResult
from celeste_search.room_planning import action_fingerprint


class Branches:
    map_data=None
    last_step_started_dash=False

    def step(self,state,mask):
        x={(0.,LEFT):20.,(0.,RIGHT):20.,(20.,RIGHT):84.}.get((state.x,mask))
        return replace(state,x=state.x if x is None else x,dead=x is None,
            frame=state.frame+1,prev_input=mask,stamina=state.stamina-2,
            vx=(-20. if mask==LEFT else 20.) if state.x==0 else state.vx)


def policy(sim,start,goal,**kwargs):
    actions=[LEFT if sim.no_dash else RIGHT] if start.x==0 else [RIGHT]
    final=sim.step(start,actions[0])
    return SearchResult(actions,final,goal.achieved(final),1,1,'fixture')


class FailedCandidateAlternativesTests(unittest.TestCase):
    def run_search(self,kind,blacklist):
        start=SimState('a',0.,8.,stamina=83.,frame=91)
        route=dict(waypoints=[dict(room='a',x=16,y=0,tile_size=8),
                              dict(room='a',x=80,y=0,tile_size=8)],
                   segments=[dict(no_dash=False,no_ground=False) for _ in range(2)])
        goals=[_selected_goal(p)[0] for p in route['waypoints']]
        sim=Branches()
        with ExitStack() as stack:
            if kind=='corridor':
                stack.enter_context(patch('celeste_search.corridor_planning.generate_corridor_routes',
                    side_effect=lambda sim,s,g,**kw:[CorridorRoute('direct','a',(),((s.x,s.y),(g.x,g.y)),(),84.)]))
                stack.enter_context(patch('celeste_search.corridor_planning.search_leg',side_effect=policy))
                solver=search_corridor_routes
            else:
                stack.enter_context(patch('celeste_search.mechanism_routes.relay_candidates',return_value=[]))
                stack.enter_context(patch('celeste_search.mechanism_routes.relay_priority',return_value=0.))
                stack.enter_context(patch('celeste_search.mechanism_planning.search_leg',side_effect=policy))
                solver=search_mechanism_routes
            answer=solver(sim,start,route,goals,deadline=time.monotonic()+1,
                          max_expansions=100,max_frames=12,forbidden=blacklist)
        if answer[2]:
            state=start;phase=start_route_progress(route,state)
            for mask in answer[0]:
                before=state;state=sim.step(before,mask)
                phase=advance_route_progress(route,phase,before,state,mask)
            self.assertEqual(state,answer[1]);self.assertTrue(route_complete(route,phase))
        return answer

    def test_complete_fingerprint_exclusion_backtracks_across_public_stage(self):
        for kind in ('corridor','mechanism'):
            with self.subTest(kind=kind):
                baseline=self.run_search(kind,())
                self.assertTrue(baseline[2]);self.assertEqual(baseline[0],[LEFT,RIGHT])
                answer=self.run_search(kind,{action_fingerprint(baseline[0])})
                self.assertTrue(answer[2]);self.assertEqual(answer[0],[RIGHT,RIGHT])
                self.assertGreater(answer[6]['excluded_candidates'],0)
                self.assertEqual(answer[1].stamina,79.)

    def test_all_known_complete_candidates_excluded_never_reports_success(self):
        for kind in ('corridor','mechanism'):
            with self.subTest(kind=kind):
                answer=self.run_search(kind,{action_fingerprint([LEFT,RIGHT]),action_fingerprint([RIGHT,RIGHT])})
                self.assertFalse(answer[2]);self.assertGreater(answer[6]['excluded_candidates'],0)


if __name__=='__main__':unittest.main()
