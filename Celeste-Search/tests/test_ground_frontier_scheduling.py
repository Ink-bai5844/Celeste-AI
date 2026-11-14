"""Same-support arrivals get a turn before their old parent exhausts its work."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal, target_reached
from celeste_search.assist_routes import (advance_route_progress, normalize_route,
                                         start_route_progress)
from celeste_search.maps import MapData, Room
from celeste_search.physics import RIGHT, Simulator
from celeste_search.planner import GroundArrival, SearchResult, ground_support_key
from celeste_search.regional_planning import search_regional_routes


def fixture():
    room=Room("a",0,0,192,160,("0"*24,)*18+("1"*24,)*2,(),((16.,144.),))
    return MapData("same-support-turns",{"a":room},Path("same-support-turns.bin"))


class GroundFrontierSchedulingTests(unittest.TestCase):
    def exercise(self, prefix, *, stamina=110.):
        data=fixture()
        simulator=Simulator(data)
        start=replace(simulator.initial_state("a"),stamina=stamina,frame=37)
        route=normalize_route(data,dict(room="a",x=80,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        calls=[]
        emitted=[]

        def controlled_search(sim, actual, target, **kwargs):
            # This wrapper controls only which already simulated candidate the
            # low-level search reports. It never fabricates a reachable state
            # or declares the deliberately unreachable public target complete.
            calls.append((actual,sim.no_dash))
            if not emitted and actual==start:
                current=actual
                for mask in prefix:
                    current=sim.step(current,mask)
                    self.assertFalse(current.dead)
                self.assertTrue(current.on_ground)
                self.assertGreaterEqual(current.vy,0.)
                self.assertEqual(ground_support_key(sim,actual),ground_support_key(sim,current))
                self.assertFalse(target_reached(current,route["waypoints"][0]))
                arrival=GroundArrival(tuple(prefix),current,ground_support_key(sim,current))
                kwargs["ground_arrivals"].append(arrival)
                emitted.append(arrival)
            return SearchResult([],actual,False,1,len(prefix) if len(calls)==1 else 0,
                                "controlled_budget")

        with patch("celeste_search.regional_planning.search_leg",side_effect=controlled_search):
            result=search_regional_routes(simulator,start,route,[goal],
                deadline=time.monotonic()+5,max_expansions=400,max_frames=240)
        self.assertEqual(len(emitted),1)
        candidate=emitted[0]
        candidate_calls=[index for index,(state,_) in enumerate(calls) if state==candidate.state]
        self.assertTrue(candidate_calls,
                        "A real changed arrival on the same floor must itself be probed")
        first_full_probe=next((index for index,(_,no_dash) in enumerate(calls) if not no_dash),len(calls))
        self.assertLess(first_full_probe,len(calls),"The controlled run must exercise the parent's later probes")
        self.assertLess(candidate_calls[0],first_full_probe,
                        "The continuation must run before the parent reaches its final full-dash probes")
        self.assertLessEqual(candidate_calls[0],3,
                             "The old parent must not monopolize this support until exhaustion")
        # Verify the exported candidate and the returned public result against
        # independent raw physics from the untouched original complete state.
        raw=Simulator(data)
        replay=start
        progress=start_route_progress(route,start)
        for mask in candidate.actions:
            before=replay
            replay=raw.step(replay,mask)
            progress=advance_route_progress(route,progress,before,replay,mask)
        self.assertEqual(replay,candidate.state)
        self.assertEqual(progress.index,0)
        self.assertIsNone(progress.violation)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,0)
        self.assertEqual(result[0],[])
        self.assertEqual(result[1],start)
        self.assertGreater(result[6]["ground_frontier"]["expanded"],0)
        self.assertLessEqual(result[6]["ground_frontier"]["max_pending_per_support"],12)
        return start,candidate,calls

    def test_same_floor_high_speed_arrival_runs_before_parent_exhaustion(self):
        start,candidate,_=self.exercise([RIGHT]*8)
        self.assertEqual(start.vx,0.)
        self.assertEqual(candidate.state.vx,90.)
        self.assertGreater(candidate.state.x,start.x)
        self.assertEqual(candidate.state.frame,start.frame+8)

    def test_same_position_restored_resource_state_gets_its_own_turn(self):
        start,candidate,_=self.exercise([0],stamina=53.)
        self.assertEqual((candidate.state.x,candidate.state.y,candidate.state.vx),
                         (start.x,start.y,start.vx))
        self.assertEqual(start.stamina,53.)
        self.assertEqual(candidate.state.stamina,110.)
        self.assertEqual(candidate.state.frame,start.frame+1)


if __name__=="__main__":
    unittest.main()
