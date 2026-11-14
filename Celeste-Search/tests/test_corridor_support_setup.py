"""Optional support preparation cannot recursively hide its actual milestone."""
from contextlib import ExitStack
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorRoute
from celeste_search.maps import MapData, Room
from celeste_search.physics import Simulator, LEFT, RIGHT, DASH
from celeste_search.planner import Goal, SearchResult


class SupportSetupTests(unittest.TestCase):
    def run_case(self, *, surfaces=(64,), blocked=False):
        room=Room("plain",0,0,320,160,("0"*40,)*18+("1"*40,)*2,(),((16.,144.),))
        data=MapData("setup-continuation",{"plain":room},Path("setup-continuation.bin"))
        sim=Simulator(data)
        start=sim.initial_state()
        route=normalize_route(data,dict(room="plain",x=176,y=136))
        goal=_selected_goal(route["waypoints"][0])[0]
        portals=tuple(Goal("surface","plain",x,144,id=f"surface:{x}") for x in surfaces)
        corridor=CorridorRoute("optional-setup","plain",portals,
            ((16.,144.),*((p.x,p.y) for p in portals),(180.,144.)),(),200.)
        calls=[]

        def roll(wrapped, origin, actions):
            state=origin
            for mask in actions:
                state=wrapped.step(state,mask)
                self.assertFalse(state.dead)
            return state

        def support(wrapped, origin, portal):
            if portal.kind!="surface":
                return ()
            calls.append(("setup_offer",portal.id,origin.frame))
            # Deliberately offer another optional setup at every arrival.
            # These are real walk prefixes; the controlled provider isolates
            # scheduling from the separate wall-contact geometry tests.
            mask=RIGHT if origin.x<32 else LEFT
            end=roll(wrapped,origin,[mask]*18)
            return (Goal("wall_grip","plain",end.x,end.y,id="optional-catch"),)

        def approach(wrapped, origin, target, **kwargs):
            calls.append(("setup_run",target.id,origin.frame))
            kwargs["stats"]["expansions"]+=1
            mask=RIGHT if target.x>origin.x else LEFT
            actions=[mask]*18
            end=roll(wrapped,origin,actions)
            return [(actions,end,"actual_walk_prefix")]

        def search(wrapped, origin, target, **kwargs):
            calls.append(("probe",target.id,origin.frame))
            actions=[]
            state=origin
            # The first tiny probe is inconclusive. A genuine setup arrival
            # makes the ordinary milestone solver eligible; it must get a
            # chance before a second setup can consume the finite budget.
            if origin!=start and not blocked:
                for _ in range(min(160,kwargs["max_frames"])):
                    if target.achieved(state):
                        break
                    mask=RIGHT if target.x>=state.x else LEFT
                    state=wrapped.step(state,mask)
                    actions.append(mask)
            return SearchResult(actions,state,target.achieved(state),1,len(actions),"frontier_exhausted")

        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]))
            stack.enter_context(patch("celeste_search.corridor_planning._support_grips",side_effect=support))
            stack.enter_context(patch("celeste_search.corridor_planning._opposite_grips",return_value=()))
            stack.enter_context(patch("celeste_search.corridor_planning.search_leg",side_effect=search))
            stack.enter_context(patch("celeste_search.wall_moves.wall_approach_edges",side_effect=approach))
            stack.enter_context(patch("celeste_search.wall_moves.wall_relay_approach_edges",return_value=[]))
            result=search_corridor_routes(sim,start,route,[goal],deadline=time.monotonic()+3.,
                max_expansions=12,max_frames=300)
        replay=start
        for mask in result[0]:
            replay=sim.step(replay,mask)
        self.assertEqual(replay,result[1])
        self.assertTrue(all(not mask&DASH for mask in result[0]))
        return result,route,calls

    def test_prepared_arrival_probes_original_surface_before_another_setup(self):
        result,route,calls=self.run_case()
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertEqual(sum(row[0]=="setup_run" for row in calls),1,calls)
        self.assertTrue(any(kind=="probe" and target=="surface:64" and frame>0
                            for kind,target,frame in calls),calls)
        self.assertGreater(result[6]["support_setup_target_attempts"],0)

    def test_reaching_the_next_milestone_enables_its_own_setup(self):
        result,route,calls=self.run_case(surfaces=(64,112))
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertEqual([target for kind,target,_ in calls if kind=="setup_offer"],
                         ["surface:64","surface:112"])
        self.assertEqual(sum(row[0]=="setup_run" for row in calls),2,calls)

    def test_setup_alone_cannot_complete_the_public_route(self):
        result,route,calls=self.run_case(blocked=True)
        self.assertFalse(result[2])
        self.assertFalse(route_complete(route,result[5]))
        # All offers remain at the original node. Prepared child states get
        # ordinary target probes and then backtrack instead of another setup.
        self.assertTrue(all(frame==0 for kind,_,frame in calls if kind=="setup_offer"),calls)
        self.assertGreater(result[6]["backtracks"],0)


if __name__=="__main__":
    unittest.main()
