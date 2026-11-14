"""Atomic mechanism recipes cannot spend the whole remaining request time."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import GRAB,UP,Simulator
from celeste_search.planner import Goal,SearchResult
from celeste_search.wall_moves import WallGripGoal


def fixture():
    room=Room("a",0,0,320,160,("0"*40,)*18+("1"*40,)*2,(),((16.,144.),))
    return MapData("atomic-macros",{"a":room},Path("atomic-macros.bin"))


def corridor(identity,portal):
    return CorridorRoute(identity,"a",(portal,),((16.,144.),(portal.x,portal.y),(284.,9.5)),(),200.)


class AtomicMacroQuantumTests(unittest.TestCase):
    def test_each_macro_family_receives_its_own_short_deadline(self):
        clock=[100.];calls=[]
        data=fixture();sim=Simulator(data);start=replace(sim.initial_state(),state=1)
        route=normalize_route(data,dict(room="a",x=280,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        wall=WallGripGoal("wall_grip","a",80,40,id="wall",rect=(84,8,92,150),facing=1)
        candidates=[corridor("wall",wall),corridor("refill",Goal("refill","a",160,40,id="refill")),
                    corridor("clearance",Goal("mechanism_clearance","a",200,8,id="clearance"))]
        def capture(name):
            def helper(*args,**kwargs):
                calls.append((name,kwargs["deadline"]-clock[0]))
                kwargs["stats"]["expansions"]+=1
                clock[0]+=.01
                return []
            return helper
        def search(wrapped,origin,target,**kwargs):
            return SearchResult([],origin,False,1,0,"frontier_exhausted")
        with patch("celeste_search.corridor_planning.time.monotonic",side_effect=lambda:clock[0]), \
             patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=candidates), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.corridor_planning._mechanism_clearance_edges",side_effect=capture("clearance")), \
             patch("celeste_search.mechanism_moves.mechanism_transfer_edges",side_effect=capture("mechanism_transfer")), \
             patch("celeste_search.wall_underpasses.wall_underpass_edges",side_effect=capture("wall_underpass")), \
             patch("celeste_search.wall_transfers.static_wall_transfer_edges",side_effect=capture("transfer")), \
             patch("celeste_search.refill_moves.refill_approach_edges",side_effect=capture("refill")), \
             patch("celeste_search.wall_moves.wall_approach_edges",side_effect=capture("wall")), \
             patch("celeste_search.wall_moves.wall_relay_approach_edges",side_effect=capture("relay")):
            result=search_corridor_routes(sim,start,route,[goal],deadline=105.,
                                         max_expansions=500,max_frames=200)
        self.assertEqual({name for name,_ in calls},{"clearance","transfer","refill","wall","relay",
                                                  "mechanism_transfer","wall_underpass"})
        self.assertTrue(all(0<=duration<=.200001 for _,duration in calls),calls)
        self.assertFalse(result[2])
        self.assertEqual(result[6]["corridor_scheduler"]["atomic_macro_seconds"],.2)

    def test_a_slow_atomic_macro_yields_to_the_next_corridor(self):
        clock=[100.];events=[]
        data=fixture();sim=Simulator(data);start=sim.initial_state()
        route=normalize_route(data,dict(room="a",x=280,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        wall=WallGripGoal("wall_grip","a",80,40,id="slow",rect=(84,8,92,150),facing=1)
        candidates=[corridor("slow",wall),corridor("next",Goal("corridor","a",180,8,id="next"))]
        def slow(*args,**kwargs):
            events.append(("macro","slow",clock[0],kwargs["deadline"]))
            kwargs["stats"]["expansions"]+=1
            clock[0]=kwargs["deadline"]  # Simulate a helper using its entire allowance.
            return []
        def search(wrapped,origin,target,**kwargs):
            events.append(("probe",target.id,clock[0],kwargs["deadline"]))
            return SearchResult([],origin,False,1,0,"frontier_exhausted")
        with patch("celeste_search.corridor_planning.time.monotonic",side_effect=lambda:clock[0]), \
             patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=candidates), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.wall_moves.wall_approach_edges",side_effect=slow), \
             patch("celeste_search.wall_moves.wall_relay_approach_edges",return_value=[]):
            result=search_corridor_routes(sim,start,route,[goal],deadline=105.,
                                         max_expansions=100,max_frames=200)
        index=next(i for i,event in enumerate(events) if event[0]=="macro")
        following=[event for event in events[index+1:] if event[:2]==("probe","next")]
        self.assertTrue(following,events)
        self.assertLessEqual(following[0][2]-events[index][2],.200001)
        self.assertLess(following[0][2],105.)
        self.assertFalse(result[2])
        self.assertEqual(result[0],[])

    def test_climb_extension_is_bounded_even_with_slow_real_physics_steps(self):
        clock=[100.];masks=[];seen=set()
        room=Room("a",0,0,192,160,("0"*24,)*20,
                  (Entity("solid",1,64,16,16,128),),((84.,100.),))
        data=MapData("slow-climb",{"a":room},Path("slow-climb.bin"))
        sim=Simulator(data)
        from celeste_search.physics import SimState
        start=SimState("a",84.,100.,state=1,facing=-1,stamina=110.)
        raw_step=sim.step
        def timed(state,mask):
            clock[0]+=.06
            # Certification replays already calculated inputs without opening
            # new branches. Count only new climb work within each scheduler
            # turn; subsequent turns must continue from the paused state.
            identity=(state,mask)
            masks.append(mask if identity not in seen else 0)
            seen.add(identity)
            return raw_step(state,mask)
        sim.step=timed
        route=normalize_route(data,dict(room="a",x=160,y=0))
        goal=_selected_goal(route["waypoints"][0])[0]
        wall=WallGripGoal("wall_grip","a",84,100,id="climb",rect=(64,16,80,144),facing=-1)
        proposal=CorridorRoute("climb","a",(wall,Goal("corridor","a",100,32)),
                               ((84.,100.),(84.,32.),(164.,9.5)),(),100.)
        def approach(wrapped,origin,target,**kwargs):
            kwargs["stats"]["expansions"]+=1
            after=wrapped.step(origin,GRAB)
            kwargs["stats"]["simulated_frames"]+=1
            return [([GRAB],after,"actual_grip")] if target.achieved(after) else []
        def search(wrapped,origin,target,**kwargs):
            return SearchResult([],origin,False,1,0,"frontier_exhausted")
        with patch("celeste_search.corridor_planning.time.monotonic",side_effect=lambda:clock[0]), \
             patch("celeste_search.corridor_planning.publish_corridor_progress",side_effect=lambda *a,**k:masks.append(0)), \
             patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[proposal]), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search), \
             patch("celeste_search.mechanism_moves.mechanism_transfer_edges",return_value=[]), \
             patch("celeste_search.wall_underpasses.wall_underpass_edges",return_value=[]), \
             patch("celeste_search.wall_transfers.static_wall_transfer_edges",return_value=[]), \
             patch("celeste_search.wall_moves.wall_approach_edges",side_effect=approach), \
             patch("celeste_search.wall_moves.wall_relay_approach_edges",return_value=[]):
            result=search_corridor_routes(sim,start,route,[goal],deadline=105.,
                                         max_expansions=200,max_frames=200)
        runs=[];run=0
        for mask in masks:
            if mask==GRAB|UP:run+=1
            elif run:runs.append(run);run=0
        if run:runs.append(run)
        self.assertTrue(runs,masks)
        self.assertLessEqual(max(runs),4,"A 0.2-second quantum permits at most four 0.06-second steps")
        self.assertGreater(sum(runs),4,"Later turns must resume the remaining climb")
        self.assertFalse(result[2])


if __name__=="__main__":unittest.main()
