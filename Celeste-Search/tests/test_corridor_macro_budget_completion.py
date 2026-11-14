"""A completed atomic recipe keeps its proof when it consumes the last budget."""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route,route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.physics import Simulator,RIGHT
from celeste_search.planner import Goal,SearchResult
from celeste_search.room_planning import action_fingerprint
from test_corridor_atomic_macro_quantum import fixture,corridor


class MacroBudgetCompletionTests(unittest.TestCase):
    def run_recipe(self, family="refill", *, target_x=40, time_limit=False,
                   partial_first=False, forbidden=(), reject_first_hit=False):
        clock=[100.]; calls=[]
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        if family=="transfer":
            start=replace(start,state=1)
        route=normalize_route(data,dict(room="a",x=target_x,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        kind={"clearance":"mechanism_clearance","transfer":"corridor",
              "wall":"wall_grip","relay":"wall_grip"}.get(family,"refill")
        portal=Goal(kind,"a",44,144,id="probe")
        def search(wrapped,origin,target,**kwargs):
            calls.append("search")
            return SearchResult([],origin,False,1,0,"frontier_exhausted")
        def macro(wrapped,origin,target,**kwargs):
            calls.append("macro")
            kwargs["stats"]["expansions"]+=1
            rows=[]
            for count in ([14,22] if partial_first else [22]):
                actions=[RIGHT]*count;state=origin
                for mask in actions:
                    state=wrapped.step(state,mask)
                self.assertFalse(state.dead)
                rows.append((actions,state,"actual"))
            if time_limit:
                clock[0]=kwargs["deadline"]
            return [row[0] for row in rows] if family=="clearance" else rows
        helpers={"clearance":"celeste_search.corridor_planning._mechanism_clearance_edges",
                 "transfer":"celeste_search.wall_transfers.static_wall_transfer_edges",
                 "refill":"celeste_search.refill_moves.refill_approach_edges",
                 "wall":"celeste_search.wall_moves.wall_approach_edges",
                 "relay":"celeste_search.wall_moves.wall_relay_approach_edges"}
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.corridor_planning.time.monotonic",side_effect=lambda:clock[0]))
            stack.enter_context(patch("celeste_search.corridor_planning.generate_corridor_routes",
                                      return_value=[corridor("only",portal)]))
            stack.enter_context(patch("celeste_search.corridor_planning.search_leg",side_effect=search))
            stack.enter_context(patch("celeste_search.corridor_planning._opposite_grips",return_value=()))
            for name,path in helpers.items():
                stack.enter_context(patch(path,side_effect=macro if name==family else None,
                                          return_value=[]))
            if reject_first_hit:
                stack.enter_context(patch.object(type(goal),"first_hit_rejected",return_value=True))
            result=search_corridor_routes(sim,start,route,[goal],deadline=100.2 if time_limit else 105.,
                max_expansions=2 if not time_limit else 20,max_frames=200,forbidden=forbidden)
        self.assertEqual(calls,["search","macro"],"Budget drain must not launch a new search or macro")
        self.assertEqual(result[3],2)
        replay=start
        for mask in result[0]:
            replay=sim.step(replay,mask)
        self.assertEqual(replay,result[1],"Only the exact replayed physical prefix may be returned")
        return result,route,goal

    def test_every_macro_family_retains_complete_answer_on_last_expansion(self):
        for family in ("clearance","transfer","refill","wall","relay"):
            with self.subTest(family=family):
                result,route,goal=self.run_recipe(family)
                self.assertTrue(result[2],result[6])
                self.assertTrue(route_complete(route,result[5]))
                self.assertTrue(goal.achieved(result[1]))

    def test_macro_finishing_on_deadline_can_certify_its_existing_answer(self):
        result,_,_=self.run_recipe(time_limit=True)
        self.assertTrue(result[2],result[6])

    def test_later_complete_result_is_not_hidden_by_private_arrival_in_same_batch(self):
        result,_,_=self.run_recipe(partial_first=True)
        self.assertTrue(result[2],result[6])

    def test_private_portal_at_last_expansion_does_not_complete_public_goal(self):
        result,route,goal=self.run_recipe(target_x=160)
        self.assertFalse(result[2])
        self.assertFalse(route_complete(route,result[5]))
        self.assertFalse(goal.achieved(result[1]))

    def test_completed_macro_still_respects_blacklist_and_first_hit_rejection(self):
        accepted,_,_=self.run_recipe()
        for kwargs in (dict(forbidden=(action_fingerprint(accepted[0]),)),dict(reject_first_hit=True)):
            with self.subTest(kwargs=kwargs):
                result,_,_=self.run_recipe(**kwargs)
                self.assertFalse(result[2])


if __name__=="__main__":unittest.main()
