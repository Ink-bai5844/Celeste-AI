"""Dynamic clearance stages advance real player and solid states together."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import start_route_progress, advance_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator, _mechanism_clearance_edges, _opposite_grips
from celeste_search.corridor_routes import FallingBlockTriggerGoal, MechanismClearanceGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import WallGripGoal


class CorridorMechanismTests(unittest.TestCase):
    def setUp(self):
        room=Room("a",0,0,160,144,("0"*20,)*16+("1"*20,)*2,
                  (Entity("fallingBlock",1,80,32,16,64),),())
        self.sim=Simulator(MapData("dynamic-clearance",{"a":room},Path("fixture.bin")))
        self.start=SimState("a",76.,96.,state=1,facing=1,dashes=1,stamina=110,
                           mechanisms=(("a:1",1,0,80.,32.,0.),))
        self.goal=MechanismClearanceGoal("mechanism_clearance","a",76.,96.,
            identity="a:1",rect=(80.,32.,96.,96.),clearance_box=(80.,32.,96.,48.))

    def edges(self,sim=None,**options):
        settings=dict(deadline=time.monotonic()+2,max_frames=150,
                      stats=dict(expansions=0,simulated_frames=0),max_expansions=12)
        settings.update(options)
        return _mechanism_clearance_edges(sim or self.sim,self.start,self.goal,**settings)

    def test_wait_requires_real_motion_and_preserves_replayable_inputs(self):
        self.assertFalse(self.goal.achieved(self.start))
        edges=self.edges()
        self.assertTrue(edges)
        self.assertEqual(self.start.mechanisms[0][4],32.)
        for actions in edges:
            self.assertGreater(len(actions),1)
            self.assertFalse(any(mask&DASH for mask in actions))
            state=self.start
            for mask in actions:
                self.assertFalse(self.goal.achieved(state))
                state=self.sim.step(state,mask)
                self.assertFalse(state.dead)
            self.assertEqual(state.frame,len(actions))
            self.assertTrue(self.goal.achieved(state))

    def test_deadline_and_expansion_exhaustion_never_advance_mechanisms(self):
        with patch.object(self.sim,"step",wraps=self.sim.step) as step:
            self.assertEqual(self.edges(deadline=time.monotonic()-1),[])
            self.assertEqual(self.edges(max_expansions=0),[])
            step.assert_not_called()
        self.assertEqual(self.edges(max_frames=1),[])

    def test_wait_does_not_bypass_public_airborne_constraint(self):
        route=dict(waypoints=[dict(room="a",x=136,y=16,tile_size=8)],
                   segments=[dict(no_dash=True,no_ground=True)])
        wrapped=_ConstrainedSimulator(self.sim,route,no_dash=True)
        for actions in self.edges(wrapped):
            state=self.start
            phase=start_route_progress(route,state)
            for mask in actions:
                before=state
                state=self.sim.step(state,mask)
                phase=advance_route_progress(route,phase,before,state,mask)
                self.assertFalse(state.dead)
                self.assertFalse(phase.violation)
            self.assertTrue(self.goal.achieved(state))

    def test_clearance_arrivals_follow_the_next_passage_without_discarding_other_heights(self):
        def arrival(actions):
            state=self.start
            for mask in actions:
                state=self.sim.step(state,mask)
            return state
        high=Goal("corridor","a",76.,75.)
        low=Goal("corridor","a",76.,128.)
        high_edges=self.edges(destination=high)
        low_edges=self.edges(destination=low)
        self.assertEqual(arrival(high_edges[0]).y,75.)
        self.assertEqual(arrival(low_edges[0]).y,128.)
        # Ranking changes which real branch is attempted first; it must not
        # erase the other resource/position states needed for backtracking.
        self.assertEqual({tuple(a) for a in high_edges},{tuple(a) for a in low_edges})

    def test_trigger_cannot_be_substituted_by_gripping_an_unrelated_opposite_wall(self):
        room=self.sim.map.rooms["a"]
        room=replace(room,entities=room.entities+(Entity("solid",2,48,0,8,128),))
        sim=Simulator(MapData("opposite-trigger",{"a":room},Path("fixture.bin")))
        static=WallGripGoal("wall_grip","a",76.,96.,rect=(80.,32.,96.,96.),facing=1)
        trigger=FallingBlockTriggerGoal("wall_grip","a",76.,96.,
            rect=(80.,32.,96.,96.),facing=1,identity="a:1",mechanism_kind="fallingBlock")
        self.assertTrue(_opposite_grips(sim,self.start,static))
        self.assertEqual(_opposite_grips(sim,self.start,trigger),())


if __name__=="__main__":
    unittest.main()
