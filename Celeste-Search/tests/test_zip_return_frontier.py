"""Real held return and bounded climbing windows survive mechanism dispatch."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete, start_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_planning import search_mechanism_routes, _zip_launch_windows
from celeste_search.mechanism_routes import relay_candidates
from celeste_search.physics import GRAB, LEFT, RIGHT, UP, Simulator, SimState
from celeste_search.planner import Goal, SearchResult
from celeste_search.zip_grips import ZipContactGoal, _contact, zip_grip_edges


def fixture(*, mirror=False, platform=False):
    x, facing = (224, 1) if mirror else (80, -1)
    entities = [Entity("zipMover", 1, x, 48, 16, 64, {}, ((x,136),)),
                Entity("spikesUp", 2, x, 48, 16, 0)]
    if platform:
        entities.append(Entity("solid", 3, x-16 if mirror else x+16, 128, 16, 16))
    room = Room("return", 0, 0, 320, 320, ("0"*40,)*40, tuple(entities), ((32.,304.),))
    sim = Simulator(MapData("return", {room.name:room}, Path("return.bin")))
    mask = GRAB|(RIGHT if mirror else LEFT)
    start = SimState(room.name, x-4 if mirror else x+20, 108, state=1,
        facing=facing, stamina=110, dashes=1, frame=163, prev_input=mask,
        mechanisms=(("return:1",1,7,float(x),48.,0.),))
    grip = ZipContactGoal("wall_grip",room.name,start.x,60,identity="return:1",
        rect=(x,48,x+16,112),facing=facing,mechanism_kind="zipMover")
    goal = Goal("room","not_reached",0,0)
    return sim,start,grip,goal,mask


class ZipReturnFrontierTests(unittest.TestCase):
    def test_first_contact_keeps_return_height_and_exact_resources_in_both_directions(self):
        for mirror in (False,True):
            sim,start,grip,goal,held = fixture(mirror=mirror)
            variants=[]
            for climb in (False,True):
                stats={}
                rows=list(zip_grip_edges(sim,start,grip,goal,deadline=time.monotonic()+1,
                    max_frames=280,stats=stats,cycle_windows=True,return_climb=climb))
                self.assertEqual(stats["expansions"],1)
                self.assertEqual(stats["simulated_frames"],256)
                returned=next(row for row in rows if row[2].endswith(":return_end"))
                self.assertEqual(returned[1].mechanisms[0][1:3],(5,30))
                self.assertTrue(any(row[2].endswith(":return72") for row in rows))
                self.assertTrue(any(":cycle1:" in row[2] for row in rows))
                self.assertTrue(all(mask == held or climb and mask == held|UP
                                    for row in rows for mask in row[0]))
                for actions,final,_ in rows:
                    state=start
                    for mask in actions:
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                        self.assertTrue(_contact(state,grip))
                    self.assertEqual(state,final)
                    self.assertLess(final.stamina,start.stamina)
                    self.assertEqual(final.dashes,start.dashes)
                variants.append(rows)
            hold_end=next(row[1] for row in variants[0] if row[2].endswith(":return_end"))
            climb_end=next(row[1] for row in variants[1] if row[2].endswith(":return_end"))
            self.assertLess(climb_end.y,hold_end.y-32)
            self.assertGreater(climb_end.stamina,0)
            selected=_zip_launch_windows(variants[0]+variants[1])
            self.assertLessEqual(len(selected),12)
            self.assertTrue(any(":return_climb:" in row[2] and row[2].endswith(":return_end") for row in selected))
            self.assertTrue(any(":return_climb:" not in row[2] and row[2].endswith(":return_end") for row in selected))

    def test_existing_contact_is_extended_once_and_return_nodes_are_actually_expanded(self):
        sim,start,grip,_,_=fixture()
        route=normalize_route(sim.map,dict(room="return",x=16,y=0))
        public=_selected_goal(route["waypoints"][0])[0]
        expanded=[]
        def contacts(wrapped,state,*args,**kwargs):
            expanded.append(state)
            # Use actual ranked inventory: it deliberately omits the already
            # achieved grip. The consumer must discover that real contact.
            result=relay_candidates(wrapped,state,*args,**kwargs)
            self.assertTrue(all(not target.achieved(state) for target in result))
            return result
        def no_suffix(wrapped,state,target,**kwargs):
            return SearchResult([],state,False,1,0,"controlled_no_public")
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",side_effect=contacts))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority",return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg",side_effect=no_suffix))
            for name in ("mechanism_moves.mechanism_transfer_edges","wall_transfers.static_wall_transfer_edges",
                         "wall_underpasses.wall_underpass_edges","wall_descents.wall_descent_edges",
                         "wall_moves.wall_approach_edges"):
                stack.enter_context(patch("celeste_search."+name,return_value=[]))
            spy=stack.enter_context(patch("celeste_search.zip_grips.zip_grip_edges",wraps=zip_grip_edges))
            result=search_mechanism_routes(sim,start,route,[public],deadline=time.monotonic()+2,
                max_expansions=500,max_frames=600,use_detours=False)
        self.assertEqual(spy.call_count,2,"Held and climb variants run once, not recursively for each window")
        self.assertTrue(any(state.mechanisms[0][1]==4 for state in expanded))
        self.assertTrue(any(state.mechanisms[0][1]==5 and state.y-state.mechanisms[0][4]<20
                            for state in expanded),"A real high return state must reach continuation dispatch")
        self.assertFalse(result[2])
        self.assertFalse(route_complete(route,result[5]))
        self.assertLessEqual(result[6]["nodes"],13)
        actual=start
        for mask in result[0]:actual=sim.step(actual,mask)
        self.assertEqual(actual,result[1])

    def test_return_policy_obeys_actual_forbidden_landing_and_budget(self):
        sim,start,grip,goal,mask=fixture(platform=True)
        route=normalize_route(sim.map,dict(waypoints=[dict(room="return",x=16,y=0)],
            segments=[dict(no_dash=True,no_ground=True)]))
        wrapped=_ConstrainedSimulator(sim,route,no_dash=True,phase=start_route_progress(route,start))
        state=start
        for tick in range(1,100):
            state=wrapped.step(state,mask)
            if state.dead:break
        self.assertTrue(state.dead)
        self.assertEqual(state.reason,"corridor_landing_forbidden")
        stats={"expansions":0,"simulated_frames":0,"max_expansions":1}
        rows=list(zip_grip_edges(wrapped,start,grip,goal,cycle_windows=True,return_climb=True,
            deadline=time.monotonic()+1,max_frames=280,stats=stats))
        self.assertTrue(all(not row[1].dead and not row[1].on_ground for row in rows))
        self.assertLessEqual(stats["simulated_frames"],tick)
        self.assertEqual(list(zip_grip_edges(wrapped,start,grip,goal,cycle_windows=True,
            deadline=time.monotonic()+1,stats=stats)),[])


if __name__=="__main__":unittest.main()
