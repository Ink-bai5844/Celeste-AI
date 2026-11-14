"""Cumulative macro batches do not repeatedly replay/publish identical inputs."""
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.physics import Simulator,RIGHT
from celeste_search.planner import Goal,SearchResult
from celeste_search import search_progress
from test_corridor_round_robin import fixture


class AdmissionReuseTests(unittest.TestCase):
    def test_duplicate_batches_publish_one_real_anchored_prefix(self):
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        actions=[RIGHT]*18;arrival=start
        for mask in actions:
            arrival=sim.step(arrival,mask)
        route=normalize_route(data,dict(room="a",x=160,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        portal=Goal("refill","a",44,144,id="contact")
        candidate=CorridorRoute("one","a",(portal,),((16.,144.),(44.,144.),(164.,144.)),(),148.)
        calls=[];remembered=[]
        def helper(wrapped,origin,target,**kwargs):
            calls.append(kwargs["remember_results"])
            handle=kwargs["continuation"]
            handle.key="controlled physical batch"
            handle.expansions+=1
            handle.done=handle.expansions==3
            kwargs["stats"]["expansions"]+=1
            return [(actions,arrival,"actual")]
        def remember(origin,end,states):
            remembered.append((origin,end,tuple(states)))
            search_progress.remember_segment(origin,end,states)
        def search(wrapped,origin,target,**kwargs):
            return SearchResult([],origin,False,1,0,"frontier_exhausted")
        with search_progress.progress_reporter(lambda packet:None,start,route), \
             patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[candidate]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.refill_moves.refill_approach_edges",side_effect=helper), \
             patch("celeste_search.corridor_planning.remember_segment",side_effect=remember):
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+2,
                                         max_expansions=100,max_frames=200)
            composed,local=search_progress._ACTIVE.get().compose(arrival,[])
        self.assertFalse(result[2],"Only the private intermediate point was reached")
        self.assertGreaterEqual(len(calls),3)
        self.assertFalse(any(calls),"The adopting caller owns the actual prefix publication")
        actual=[row for row in remembered if row[0]==start and row[1]==arrival]
        self.assertEqual(len(actual),1)
        self.assertEqual(len(actual[0][2]),len(actions)+1)
        self.assertFalse(local)
        self.assertEqual(composed[0]["x"],start.x)
        self.assertEqual(composed[-1]["x"],arrival.x)


if __name__=="__main__":unittest.main()
