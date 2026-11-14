"""Corridor probes rotate without discarding their physical-search frontier."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import advance_route_progress,normalize_route,start_route_progress
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute,CorridorCandidates
from celeste_search.maps import Room,MapData
from celeste_search.physics import RIGHT,Simulator
from celeste_search.planner import (Goal,SearchResult,SearchContinuation,SearchNodeBudget,
                                    search_leg,_SEARCH_SESSIONS)


def fixture():
    room=Room("a",0,0,320,160,("0"*40,)*18+("1"*40,)*2,(),((16.,144.),))
    return MapData("round-robin",{"a":room},Path("round-robin.bin"))


def corridors(count):
    return [CorridorRoute(f"route-{index}","a",
        (Goal("corridor","a",40.+index*8,144.,id=f"portal-{index}"),),
        ((16.,144.),(40.+index*8,144.),(164.,9.5)),(),100.) for index in range(count)]


class CorridorRoundRobinTests(unittest.TestCase):
    def run_controlled(self,count=6,expansions=500):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(room="a",x=160,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        candidates=corridors(count)
        calls=[];pages=[]
        def generate(*args,**kwargs):
            offset=kwargs.get("candidate_offset",0)
            pages.append(offset)
            return candidates[offset:offset+kwargs["max_routes"]]
        def probe(sim,origin,target,**kwargs):
            calls.append((target.id,kwargs["max_expansions"],id(sim),id(kwargs["continuation"])))
            return SearchResult([],origin,False,kwargs["max_expansions"],0,"budget")
        with patch("celeste_search.corridor_planning.generate_corridor_routes",side_effect=generate), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=probe):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+3,
                max_expansions=expansions,max_frames=300)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,0)
        self.assertEqual(result[0],[])
        self.assertEqual(result[1],start)
        self.assertLessEqual(result[3],expansions)
        return result,calls,pages

    def test_all_six_routes_receive_a_real_probe_before_deeper_search(self):
        result,calls,_=self.run_controlled()
        self.assertEqual([row[0] for row in calls[:6]],[f"portal-{i}" for i in range(6)])
        self.assertTrue(all(row[1]<=4 for row in calls[:6]))
        self.assertEqual([row[0] for row in calls[6:12]],[f"portal-{i}" for i in range(6)])
        self.assertTrue(all(row[1]<=64 for row in calls))
        self.assertTrue(all(row["turns"]>=1 for row in result[6]["topology"]["corridors"]))
        for index in range(6):
            matching=[row for row in calls if row[0]==f"portal-{index}"]
            self.assertEqual(matching[0][2:],matching[1][2:],"Primer and next slice resume the same wrapper/handle")

    def test_later_pages_are_probed_before_first_page_spends_full_recipes(self):
        result,calls,pages=self.run_controlled(count=24,expansions=300)
        self.assertEqual(pages,[0,6,12,18])
        self.assertEqual([row[0] for row in calls[:24]],[f"portal-{i}" for i in range(24)])
        self.assertTrue(all(row[1]<=4 for row in calls[:24]))
        inventory=result[6]["topology"]["corridors"]
        self.assertEqual(len(inventory),24)
        self.assertTrue(all(row["turns"]>=1 for row in inventory))
        self.assertEqual(result[6]["corridor_scheduler"]["routes_started"],24)

    def test_actual_saved_search_resumes_after_global_lru_is_evicted(self):
        data=fixture();sim=Simulator(data)
        start=replace(sim.initial_state(),frame=77,stamina=53.,rem_x=.12)
        goal=Goal("target","a",280,144)
        handle=SearchContinuation()
        try:
            with patch("celeste_search.planner.macros",return_value=[(RIGHT,6)]):
                first=search_leg(sim,start,goal,deadline=time.monotonic()+3,max_expansions=2,
                    max_frames=100,continuation=handle)
                _SEARCH_SESSIONS.clear()  # Simulate every opportunistic entry being evicted.
                second=search_leg(sim,start,goal,deadline=time.monotonic()+3,max_expansions=3,
                    max_frames=100,continuation=handle)
            self.assertGreater(len(second.actions),len(first.actions))
            self.assertEqual(handle.resumes,1)
            self.assertEqual(second.expansions,3)
            state=start
            raw=Simulator(data)
            for mask in second.actions:
                state=raw.step(state,mask)
            self.assertEqual(state,second.state)
            self.assertEqual(state.frame,77+len(second.actions))
            self.assertFalse(second.reached)
        finally:
            handle.release()
        self.assertEqual(handle.node_budget.nodes,0)

    def test_shared_node_limit_releases_capacity_and_another_recipe_progresses(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        goal=Goal("target","a",280,144)
        pool=SearchNodeBudget(max_nodes=6)
        first,second=SearchContinuation(pool),SearchContinuation(pool)
        try:
            with patch("celeste_search.planner.macros",return_value=[(RIGHT,6)]):
                blocked=search_leg(sim,start,goal,deadline=time.monotonic()+3,
                    max_expansions=20,max_frames=100,continuation=first)
                self.assertEqual(blocked.reason,"continuation_memory_budget")
                self.assertEqual(pool.nodes,6)
                waiting=search_leg(sim,start,goal,deadline=time.monotonic()+3,
                    max_expansions=3,max_frames=100,continuation=second)
                self.assertEqual(waiting.reason,"continuation_memory_budget")
                self.assertEqual(waiting.expansions,0)
                self.assertLessEqual(pool.nodes,6)
                first.release()
                progressed=search_leg(sim,start,goal,deadline=time.monotonic()+3,
                    max_expansions=3,max_frames=100,continuation=second)
                self.assertEqual(progressed.expansions,3)
                self.assertTrue(progressed.actions)
                self.assertLessEqual(pool.nodes,6)
                state=start
                for mask in progressed.actions:state=sim.step(state,mask)
                self.assertEqual(state,progressed.state)
        finally:
            first.release();second.release()
        self.assertEqual(pool.nodes,0)

    def test_complete_starter_solution_survives_a_one_expansion_total_budget(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        for _ in range(4):start=sim.step(start,RIGHT)
        route=normalize_route(data,dict(room="a",x=24,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        near=CorridorRoute("near","a",(),((start.x,start.y),(28.,144.)),(),9.)
        with patch("celeste_search.corridor_planning.generate_corridor_routes",
                   return_value=[near]+corridors(5)) as generate:
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+2,
                                         max_expansions=1,max_frames=100)
        self.assertTrue(result[2],result[6])
        self.assertTrue(result[0])
        self.assertEqual(generate.call_count,1)
        self.assertEqual(result[6]["corridor_scheduler"]["routes_started"],1)
        current=start;raw=Simulator(data)
        for mask in result[0]:current=raw.step(current,mask)
        self.assertEqual(current,result[1])
        self.assertTrue(goal.achieved(current))

    def test_empty_timed_out_generation_retries_the_same_cursor(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        for _ in range(4):start=sim.step(start,RIGHT)
        route=normalize_route(data,dict(room="a",x=24,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        near=CorridorRoute("near","a",(),((start.x,start.y),(28.,144.)),(),9.)
        pages=[CorridorCandidates(complete=False,next_offset=0) for _ in range(4)]
        pages.append(CorridorCandidates([near],complete=True,next_offset=1,pool_size=1))
        with patch("celeste_search.corridor_planning.generate_corridor_routes",side_effect=pages) as generate:
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+2,
                                         max_expansions=5,max_frames=100)
        self.assertTrue(result[2],result[6])
        self.assertEqual([call.kwargs["candidate_offset"] for call in generate.call_args_list],[0]*5)

    def test_cancellation_releases_every_owned_search_node(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(room="a",x=160,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        pool=SearchNodeBudget(max_nodes=20)
        def cancelled(*args):raise RuntimeError("cancelled")
        sim.step=cancelled
        with patch("celeste_search.corridor_planning.SearchNodeBudget",return_value=pool), \
             patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=corridors(1)):
            with self.assertRaisesRegex(RuntimeError,"cancelled"):
                search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+2,
                                       max_expansions=10,max_frames=100)
        self.assertEqual(pool.nodes,0)

    def test_ninth_corridor_wins_with_real_physics_before_earlier_routes_exhaust_budget(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(room="a",x=160,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        candidates=[]
        for index in range(8):
            portal=Goal("corridor","a",40+index*8,8,id=f"high-{index}")
            candidates.append(CorridorRoute(f"high-{index}","a",(portal,),
                ((16.,144.),(portal.x,8.),(164.,144.)),(),250.))
        candidates.append(CorridorRoute("late-floor","a",(Goal("surface","a",80,144),),
            ((16.,144.),(80.,144.),(164.,144.)),(),148.))
        def generate(*args,**kwargs):
            offset=kwargs.get("candidate_offset",0)
            return candidates[offset:offset+kwargs["max_routes"]]
        with patch("celeste_search.corridor_planning.generate_corridor_routes",side_effect=generate):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+4,
                                         max_expansions=250,max_frames=200)
        self.assertTrue(result[2],result[6])
        self.assertEqual(result[6]["selected_corridor_id"],"late-floor")
        self.assertEqual(result[6]["corridor_scheduler"]["routes_started"],9)
        self.assertTrue(all(row["turns"]>=1 for row in result[6]["topology"]["corridors"]))
        current=start;raw=Simulator(data);phase=start_route_progress(route,start)
        for mask in result[0]:
            before=current;current=raw.step(current,mask)
            phase=advance_route_progress(route,phase,before,current,mask)
            self.assertFalse(current.dead)
            self.assertIsNone(phase.violation)
        self.assertEqual(current,result[1])
        self.assertEqual(phase,result[5])
        self.assertTrue(goal.achieved(current))
        self.assertLessEqual(result[3],250)

    def test_real_route_result_still_replays_from_exact_original_state(self):
        data=fixture();sim=Simulator(data)
        start=replace(sim.initial_state(),frame=47,stamina=72.,rem_x=.2)
        route=normalize_route(data,dict(room="a",x=160,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        corridor=CorridorRoute("physical-floor","a",(Goal("surface","a",72,144),),
                               ((16.,144.),(72.,144.),(164.,144.)),(),148.)
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+4,
                                         max_expansions=1800,max_frames=300)
        self.assertTrue(result[2],result[6])
        current=start;phase=start_route_progress(route,start);raw=Simulator(data)
        for mask in result[0]:
            before=current;current=raw.step(current,mask)
            phase=advance_route_progress(route,phase,before,current,mask)
            self.assertFalse(current.dead)
            self.assertIsNone(phase.violation)
        self.assertEqual(current,result[1])
        self.assertEqual(phase,result[5])
        self.assertEqual(phase.index,1)
        self.assertEqual(current.frame,47+len(result[0]))


if __name__=="__main__":unittest.main()
