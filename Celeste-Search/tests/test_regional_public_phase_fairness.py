"""A difficult public suffix must leave turns for better earlier arrivals."""
from dataclasses import replace
import heapq
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import (
    advance_route_progress, normalize_route, start_route_progress,
)
from celeste_search.physics import DASH, Simulator
from celeste_search.planner import SearchResult
from celeste_search import regional_planning as regional
from test_corridor_round_robin import fixture


class PublicPhaseQueueTests(unittest.TestCase):
    def test_continuously_ready_deep_phase_keeps_oldest_other_stage_turns(self):
        queue=[(-2,3,(2,"deep")),(-1,2,(1,"middle")),(0,1,(0,"root"))]
        heapq.heapify(queue)
        selected=[]
        serial=3
        for turn in range(12):
            priority,_,key=regional._pop_stage_support(queue,turn)
            selected.append(key)
            serial+=1
            heapq.heappush(queue,(priority,serial,key))
        self.assertEqual([key[0] for key in selected[:4]],[2,2,2,0])
        self.assertEqual([key[0] for key in selected[4:8]],[2,2,2,1])
        self.assertEqual(selected[11],(0,"root"))

    def test_same_stage_supports_keep_fifo_even_on_exploration_turn(self):
        queue=[(-1,serial,(1,name)) for serial,name in enumerate(("a","b","c"),1)]
        heapq.heapify(queue)
        selected=[]
        for turn in range(9):
            priority,_,key=regional._pop_stage_support(queue,turn)
            selected.append(key[1])
            heapq.heappush(queue,(priority,turn+4,key))
        self.assertEqual(selected,["a","b","c"]*3)


class PublicPhasePhysicsTests(unittest.TestCase):
    def exercise(self, *, fair=True, no_ground=False):
        data=fixture()
        simulator=Simulator(data)
        start=replace(simulator.initial_state(),frame=73,stamina=53.,rem_x=.2)
        route=normalize_route(data,dict(
            waypoints=[dict(room="a",x=40,y=136),dict(room="a",x=280,y=0)],
            segments=[dict(no_dash=True,no_ground=False),
                      dict(no_dash=True,no_ground=no_ground)]))
        goals=[_selected_goal(point)[0] for point in route["waypoints"]]
        turns=[]
        pop=regional._pop_stage_support

        def select(queue,turn):
            row=pop(queue,turn) if fair else heapq.heappop(queue)
            turns.append(row[2][0])
            return row

        def probe(wrapped,actual,target,**kwargs):
            self.assertTrue(wrapped.no_dash)
            # The unreachable high target remains unsolved. This mock charges
            # a bounded search quantum; every published arrival still comes
            # from the real local walk/jump simulation and exact replay.
            return SearchResult([],actual,False,kwargs["max_expansions"],0,"controlled_budget")

        with patch.object(regional,"_pop_stage_support",side_effect=select), \
             patch.object(regional,"search_leg",side_effect=probe):
            result=regional.search_regional_routes(simulator,start,route,goals,
                deadline=time.monotonic()+3,max_expansions=800,max_frames=180)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,1)
        self.assertLessEqual(result[3],800)
        raw=Simulator(data)
        state=start
        phase=start_route_progress(route,start)
        for mask in result[0]:
            self.assertFalse(mask&DASH)
            before=state
            state=raw.step(state,mask)
            phase=advance_route_progress(route,phase,before,state,mask)
            self.assertFalse(state.dead)
            self.assertIsNone(phase.violation)
        self.assertEqual(state,result[1])
        self.assertEqual(phase,result[5])
        self.assertEqual(state.frame,start.frame+len(result[0]))
        self.assertEqual((start.stamina,start.rem_x),(53.,.2))
        return turns,result

    def test_real_first_hit_no_longer_starves_its_pending_parent(self):
        old,_=self.exercise(fair=False)
        old_first=old.index(1)
        self.assertNotIn(0,old[old_first+1:],
                         "Control must reproduce strict public-stage starvation")
        turns,result=self.exercise()
        first=turns.index(1)
        revisited=next(index for index in range(first+1,len(turns)) if turns[index]==0)
        self.assertLessEqual(revisited-first,4)
        self.assertGreater(result[6]["ground_frontier"]["resumed"],0)

    def test_exploration_turn_preserves_hard_airborne_suffix_constraint(self):
        turns,_=self.exercise(no_ground=True)
        first=turns.index(1)
        self.assertIn(0,turns[first+1:])


if __name__=="__main__":
    unittest.main()
