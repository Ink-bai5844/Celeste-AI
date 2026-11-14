"""Real public hits earn suffix time without starving uncompleted corridors."""
from collections import Counter
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route,route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import Simulator,RIGHT
from celeste_search.planner import Goal,SearchResult
from test_corridor_round_robin import fixture


class PublicProgressFairnessTests(unittest.TestCase):
    def test_all_paths_get_first_probe_then_live_suffix_gets_extra_turns(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(waypoints=[dict(room="a",x=40,y=136),
                                                  dict(room="a",x=160,y=136)]))
        goals=[_selected_goal(point)[0] for point in route["waypoints"]]
        roots=["advanced","slow-a","slow-b"]
        probes=[];active=[];suffix_calls=0
        def corridors(simulator,origin,goal,**kwargs):
            names=roots if goal==goals[0] else ["suffix-a","suffix-b"]
            return [CorridorRoute(name,"a",(Goal("corridor","a",40 if name=="advanced" else 200,
                                                144 if name=="advanced" else 16,id=name),),
                ((origin.x,origin.y),(200.,16.)),(),100.) for name in names]
        def search(wrapped,origin,target,**kwargs):
            nonlocal suffix_calls
            phase=wrapped.phase.index
            probes.append((phase,target.id))
            suffix_calls+=int(phase==1)
            if (phase==0 and target.id=="advanced") or (phase==1 and suffix_calls>=80):
                current=origin;actions=[]
                for _ in range(150):
                    current=wrapped.step(current,RIGHT);actions.append(RIGHT)
                    if target.public_goal.achieved(current):
                        return SearchResult(actions,current,True,1,len(actions),"actual")
            return SearchResult([],origin,False,1,0,"budget")
        def publish(rows,active_id=None,**kwargs):
            if suffix_calls and active_id in roots:
                active.append(active_id)
        with patch("celeste_search.corridor_planning.generate_corridor_routes",side_effect=corridors), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.corridor_planning.publish_corridor_progress",side_effect=publish):
            result=search_corridor_routes(sim,start,route,goals,deadline=time.monotonic()+3,
                                         max_expansions=500,max_frames=240)
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        first_suffix=next(i for i,row in enumerate(probes) if row[0]==1)
        self.assertTrue(set(roots)<={name for phase,name in probes[:first_suffix] if phase==0})
        counts=Counter(active)
        self.assertGreater(counts["advanced"],2*counts["slow-a"])
        self.assertGreater(counts["slow-a"],0)
        self.assertGreater(counts["slow-b"],0)
        current=start
        for mask in result[0]:
            current=sim.step(current,mask)
            self.assertFalse(current.dead)
        self.assertEqual(current,result[1])
        self.assertTrue(goals[-1].achieved(current))

    def test_actual_private_milestones_receive_time_but_unproven_path_keeps_turns(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(room="a",x=200,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        advanced=CorridorRoute("advanced","a",tuple(Goal("corridor","a",x,144,id=f"step-{i}")
            for i,x in enumerate((48,80,120,160))),((16.,144.),(204.,144.)),(),188.)
        unproven=CorridorRoute("unproven","a",(Goal("corridor","a",280,16,id="unproven"),),
            ((16.,144.),(280.,16.),(204.,144.)),(),300.)
        probes=[];active=[];deep_calls=0
        def search(wrapped,origin,target,**kwargs):
            nonlocal deep_calls
            probes.append(target.id)
            if target.id=="step-2":
                deep_calls+=1
            if target.id in {"step-0","step-1"} or (target.id=="step-2" and deep_calls>=40):
                current=origin;actions=[]
                for _ in range(180):
                    current=wrapped.step(current,RIGHT);actions.append(RIGHT)
                    hit=(target.public_goal.achieved(current) if target.id=="step-2"
                         else target.portal.achieved(current))
                    if hit:
                        return SearchResult(actions,current,True,1,len(actions),"actual")
            return SearchResult([],origin,False,1,0,"budget")
        def publish(rows,active_id=None,**kwargs):
            if deep_calls and active_id:
                active.append(active_id)
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[advanced,unproven]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.corridor_planning.publish_corridor_progress",side_effect=publish):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+3,
                                         max_expansions=500,max_frames=240)
        self.assertTrue(result[2],result[6])
        self.assertEqual(probes[:2],["step-0","unproven"])
        counts=Counter(active)
        self.assertGreater(counts["advanced"],2*counts["unproven"])
        self.assertGreater(counts["unproven"],0)
        current=start
        for mask in result[0]:
            current=sim.step(current,mask)
            self.assertFalse(current.dead)
        self.assertEqual(current,result[1])
        self.assertTrue(goal.achieved(current))

    def test_ordered_request_keeps_fifo_for_private_progress_even_in_its_last_stage(self):
        for already_at_first in (False,True):
            with self.subTest(already_at_first=already_at_first):
                data=fixture();sim=Simulator(data);start=sim.initial_state()
                xs=(16,200) if already_at_first else (200,280)
                route=normalize_route(data,dict(waypoints=[dict(room="a",x=x,y=136) for x in xs]))
                goals=[_selected_goal(point)[0] for point in route["waypoints"]]
                advanced=CorridorRoute("advanced","a",tuple(Goal("corridor","a",x,144,id=f"step-{i}")
                    for i,x in enumerate((48,80,120,160))),((16.,144.),(204.,144.)),(),188.)
                unproven=CorridorRoute("unproven","a",(Goal("corridor","a",280,16,id="unproven"),),
                    ((16.,144.),(280.,16.),(204.,144.)),(),300.)
                probes=[];active=[];deep_calls=0
                def search(wrapped,origin,target,**kwargs):
                    nonlocal deep_calls
                    probes.append(target.id)
                    deep_calls+=int(target.id=="step-2")
                    if target.id in {"step-0","step-1"}:
                        current=origin;actions=[]
                        for _ in range(80):
                            current=wrapped.step(current,RIGHT);actions.append(RIGHT)
                            if target.portal.achieved(current):
                                return SearchResult(actions,current,True,1,len(actions),"actual")
                    return SearchResult([],origin,False,1,0,"budget")
                def publish(rows,active_id=None,**kwargs):
                    if deep_calls and active_id:
                        active.append(active_id)
                with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[advanced,unproven]), \
                     patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
                     patch("celeste_search.corridor_planning.publish_corridor_progress",side_effect=publish):
                    result=search_corridor_routes(sim,start,route,goals,deadline=time.monotonic()+3,
                                                 max_expansions=120,max_frames=240)
                self.assertFalse(result[2],"Private arrivals cannot complete the ordered public route")
                self.assertEqual(result[5].index,int(already_at_first))
                self.assertEqual(probes[:2],["step-0","unproven"])
                counts=Counter(active)
                self.assertGreater(counts["advanced"],8)
                self.assertGreater(counts["unproven"],8)
                self.assertLessEqual(abs(counts["advanced"]-counts["unproven"]),2,
                    "Private progress must not bias either stage of a multi-point request")


if __name__=="__main__":unittest.main()
