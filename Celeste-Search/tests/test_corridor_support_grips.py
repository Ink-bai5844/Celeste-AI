"""A low approach may catch a real neighboring wall before reaching a support."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.corridor_planning import (_ConstrainedSimulator, _support_grips,
                                             search_corridor_routes)
from celeste_search.corridor_routes import CorridorPortalGoal, CorridorRoute, RefillPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.planner import SearchResult
from celeste_search.wall_moves import WallGripGoal, wall_approach_edges


def fixture(*, mirror=False, offset=(0,0), dynamic=False):
    ox, oy = offset
    def x(value):
        return ox + (240-value if mirror else value)
    rows = tuple("0"*19+"1"*11 if 15<=y<26 else "0"*30 for y in range(28))
    if mirror:
        rows = tuple(row[::-1] for row in rows)
    entities = [Entity("jumpThru",1,x(152) if mirror else x(136),oy+120,16,0)]
    if dynamic:
        rows = ("0"*30,)*28
        entities.append(Entity("fallingBlock",2,x(240) if mirror else x(152),oy+120,88,88))
    room = Room("ledge",ox,oy,240,224,rows,tuple(entities),())
    sim = Simulator(MapData("support-grips",{room.name:room},Path("fixture.bin")))
    start = SimState(room.name,x(45),oy+101,vx=-151.3333 if mirror else 151.3333,
                     vy=15,dashes=1)
    px = x(144)
    portal = CorridorPortalGoal("surface",room.name,px,oy+120,
        bounds=(px-6,oy+117,px+6,oy+123),role="support")
    return sim,start,portal


class CorridorSupportGripTests(unittest.TestCase):
    def test_selected_setup_is_an_actual_exposed_wall_in_both_directions(self):
        for mirror,offset in ((False,(0,0)),(True,(-480,-800))):
            sim,start,portal=fixture(mirror=mirror,offset=offset)
            before=(start,portal)
            grips=_support_grips(sim,start,portal)
            self.assertTrue(grips)
            self.assertLessEqual(len(grips),2)
            self.assertEqual((start,portal),before)
            for grip in grips:
                self.assertEqual(grip.kind,"wall_grip")
                self.assertFalse(grip.identity)
                self.assertLessEqual(abs(grip.x-portal.x),24)
                self.assertLessEqual(abs(grip.rect[1]-portal.y),24)
                self.assertEqual(grip.facing,-1 if mirror else 1)
                # A real side collision: moving one pixel into the exposed
                # face collides, while the intended eight-pixel body is clear.
                self.assertFalse(sim.solid_at(start.room,grip.x,grip.y,state=start))
                self.assertTrue(sim.solid_at(start.room,grip.x+grip.facing,grip.y,state=start))

    def test_real_inputs_catch_wall_without_claiming_the_support_was_reached(self):
        sim,start,portal=fixture()
        grip=_support_grips(sim,start,portal)[0]
        edges=wall_approach_edges(sim,start,grip,deadline=time.monotonic()+1,max_edges=2)
        self.assertTrue(edges)
        for actions,expected,_ in edges:
            state=start
            for mask in actions:
                state=sim.step(state,mask)
                self.assertFalse(state.dead)
            self.assertEqual(state,expected)
            self.assertTrue(grip.achieved(state))
            self.assertFalse(portal.achieved(state))
            self.assertEqual(state.state,1)
            # The setup must not recursively reselect itself after catching.
            self.assertEqual(_support_grips(sim,state,portal),())

    def test_non_surface_distant_and_identity_backed_walls_are_excluded(self):
        sim,start,portal=fixture()
        for kind in ("refill","corridor","wall_grip","target"):
            self.assertEqual(_support_grips(sim,start,replace(portal,kind=kind)),())
        self.assertEqual(_support_grips(sim,start,replace(portal,x=40)),())
        self.assertEqual(_support_grips(sim,start,replace(portal,y=40)),())
        sim,start,portal=fixture(dynamic=True)
        self.assertEqual(_support_grips(sim,start,portal),())

    def test_already_reached_support_does_not_add_a_detour(self):
        sim,start,portal=fixture()
        standing=replace(start,x=portal.x,y=portal.y,on_ground=True,vy=0)
        self.assertTrue(portal.achieved(standing))
        self.assertEqual(_support_grips(sim,standing,portal),())

    def test_grounded_walk_keeps_zero_dash_route_before_optional_wall_setup(self):
        sim,original,portal=fixture()
        start=replace(original,x=200.,y=120.,vx=0.,vy=0.,on_ground=True)
        route=dict(waypoints=[dict(room=start.room,x=136,y=112,tile_size=8)],
                   segments=[dict(no_dash=False,no_ground=False)])
        goal=_selected_goal(route["waypoints"][0])[0]
        corridor=CorridorRoute("walk-first",start.room,(portal,),
            ((start.x,start.y),(portal.x,portal.y)),(),56.)
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]), \
             patch("celeste_search.wall_moves.wall_approach_edges",
                   side_effect=AssertionError("walking must not start a wall detour")):
            result=search_corridor_routes(sim,start,route,[goal],
                deadline=time.monotonic()+1,max_expansions=200,max_frames=100)
        self.assertTrue(result[2])
        self.assertTrue(result[0])
        self.assertFalse(any(mask & DASH for mask in result[0]))
        self.assertTrue(goal.achieved(result[1]))

    def test_setup_macros_keep_deadline_expansion_and_public_constraints(self):
        sim,start,portal=fixture()
        grip=_support_grips(sim,start,portal)[0]
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(wall_approach_edges(sim,start,grip,
                deadline=time.monotonic()-1),[])
            self.assertEqual(wall_approach_edges(sim,start,grip,
                deadline=time.monotonic()+1,stats={"expansions":5},max_expansions=5),[])
            step.assert_not_called()
        route=dict(waypoints=[dict(room=start.room,x=200,y=104,tile_size=8)],
                   segments=[dict(no_dash=True,no_ground=False)])
        constrained=_ConstrainedSimulator(sim,route,no_dash=True)
        self.assertEqual(wall_approach_edges(constrained,start,grip,
            deadline=time.monotonic()+1,max_expansions=32),[])


class DescendingRefillChoiceTests(unittest.TestCase):
    """A costly optional wall cannot consume the direct falling pickup's turn."""
    def solve(self, *, wall_identity="", finish=True):
        class Sim:
            map_data=None
            last_step_started_dash=False
            def step(self, state, mask):
                fields=dict(frame=state.frame+1,prev_input=mask)
                if state.x==0 and mask==LEFT:
                    return replace(state,x=20.,state=1,**fields)
                if state.x==0 and mask==RIGHT:
                    return replace(state,x=40.,y=80.,refill_timers=(("a:9",150),),**fields)
                if state.x==40 and mask==UP and finish:
                    return replace(state,x=84.,y=8.,**fields)
                return replace(state,dead=True,**fields)
        start=SimState("a",0.,40.)
        wall=WallGripGoal("wall_grip","a",20.,40.,rect=(24,0,32,96),
                          identity=wall_identity)
        refill=RefillPortalGoal("refill","a",40.,80.,identity="a:9",
                               pickup_rect=(32,66,48,82),role="refill")
        corridor=CorridorRoute("descending","a",(wall,refill),
                              ((0.,40.),(20.,40.),(40.,80.),(84.,8.)),(),100.)
        route=dict(waypoints=[dict(room="a",x=80,y=0,tile_size=8)],
                   segments=[dict(no_dash=False,no_ground=False)])
        goal=_selected_goal(route["waypoints"][0])[0]
        probes=[]
        def macro(wrapped, origin, target, *, stats, **kwargs):
            probes.append((origin.x,target.kind))
            stats["expansions"]+=1
            if origin.x!=0:
                return []
            mask=RIGHT if target.kind=="refill" else LEFT
            after=wrapped.step(origin,mask)
            stats["simulated_frames"]+=1
            return [([mask],after,"fixture")] if target.achieved(after) else []
        def search(wrapped, origin, target, **kwargs):
            actions=[UP] if origin.x==40 and finish else []
            state=origin
            for mask in actions:
                state=wrapped.step(state,mask)
            reached=bool(actions and target.achieved(state) and not state.dead)
            return SearchResult(actions,state,reached,1 if reached else kwargs["max_expansions"],
                                len(actions),"fixture")
        with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]), \
             patch("celeste_search.wall_moves.wall_approach_edges",side_effect=macro), \
             patch("celeste_search.refill_moves.refill_approach_edges",side_effect=macro), \
             patch("celeste_search.corridor_planning.search_leg",side_effect=search):
            result=search_corridor_routes(Sim(),start,route,[goal],
                deadline=time.monotonic()+1,max_expansions=24,max_frames=12)
        return result,probes

    def test_actual_descending_refill_precedes_optional_static_wall(self):
        result,probes=self.solve()
        self.assertEqual(probes[0],(0.,"refill"))
        self.assertTrue(result[2])
        self.assertEqual(result[0],[RIGHT,UP])
        self.assertEqual(result[5].index,1)
        self.assertLess(result[3],24)

    def test_dynamic_wall_is_not_reordered_by_static_wall_optimization(self):
        _,probes=self.solve(wall_identity="a:7")
        self.assertEqual(probes[0],(0.,"wall_grip"))

    def test_private_refill_contact_does_not_complete_public_goal(self):
        result,probes=self.solve(finish=False)
        self.assertEqual(probes[0],(0.,"refill"))
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,0)
        self.assertIsNone(result[6]["selected_corridor_id"])


if __name__ == "__main__":
    unittest.main()
