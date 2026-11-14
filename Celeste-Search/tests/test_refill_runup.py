"""Real ledge run-ups bridge an airborne refill beyond stationary reach."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import normalize_route, start_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.corridor_routes import RefillPortalGoal
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, LEFT, RIGHT, SimState, Simulator
import celeste_search.refill_moves as refill_moves


def runup_fixture(mirror=False, offset=(0,0), name="gap"):
    ox,oy=offset
    def px(x):return ox+(320-x if mirror else x)
    tiles=tuple("".join("1" if ty>=24 and (39-tx if mirror else tx)<13 else "0"
                        for tx in range(40)) for ty in range(32))
    entities=(Entity("refill",1,px(176),oy+160,0,0),
              Entity("refill",2,px(280),oy+20,0,0))
    room=Room(name,ox,oy,320,256,tiles,entities,((px(24),oy+192),))
    sim=Simulator(MapData("runup-gap",{name:room},Path("fixture.bin")))
    start=SimState(name,px(24),oy+192,on_ground=True,grace=6,stamina=83,
        dashes=1,frame=251,facing=-1 if mirror else 1,rem_x=-.125 if mirror else .125,
        refill_timers=((name+":2",90),),broken=frozenset({name+":prior"}))
    goal=RefillPortalGoal("refill",name,px(176),oy+165.5,identity=name+":1",
        pickup_rect=(px(176)-8,oy+152,px(176)+8,oy+168),role="refill")
    return sim,start,goal


class RefillRunupTests(unittest.TestCase):
    def test_runup_reaches_gap_stationary_family_cannot_and_replays_in_both_directions(self):
        for mirror,offset,name in ((False,(0,0),"gap"),(True,(-640,-912),"shifted")):
            with self.subTest(mirror=mirror):
                sim,start,goal=runup_fixture(mirror,offset,name)
                original=start.to_dict()
                recipes=refill_moves._recipes
                def stationary(*args):
                    return (row for row in recipes(*args)
                            if not row[1].startswith("runup_jump_dash:"))
                old_stats={}
                with patch.object(refill_moves,"_recipes",side_effect=stationary):
                    old=refill_moves.refill_approach_edges(sim,start,goal,
                        deadline=time.monotonic()+2,max_expansions=256,stats=old_stats)
                self.assertEqual(old,[],"Earlier coast/jump/dash families cannot reach this pickup")
                stats={}
                edges=refill_moves.refill_approach_edges(sim,start,goal,
                    deadline=time.monotonic()+2,max_expansions=256,stats=stats)
                self.assertTrue(edges)
                self.assertLessEqual(stats["expansions"],256)
                for actions,final,label in edges:
                    self.assertIn(":runup_jump_dash:",label)
                    runup=int(label.split(":")[2])
                    toward=LEFT if mirror else RIGHT
                    self.assertEqual(actions[:runup],[toward]*runup)
                    state=start;dashes=0
                    for index,mask in enumerate(actions,1):
                        state=sim.step(state,mask)
                        dashes+=bool(sim.last_step_started_dash)
                        self.assertFalse(state.dead)
                        if index<=runup:
                            self.assertTrue(state.on_ground)
                            self.assertEqual(state.dashes,start.dashes)
                    self.assertEqual(state,final)
                    self.assertTrue(goal.achieved(final))
                    self.assertEqual(final.frame,start.frame+len(actions))
                    self.assertEqual(final.broken,start.broken)
                    self.assertEqual(dict(final.refill_timers).get(name+":2",0),
                                     max(0,90-len(actions)))
                    self.assertGreater(dict(final.refill_timers)[goal.identity],0)
                    self.assertEqual((dashes,final.dashes),(1,1),
                                     "The actual pickup restores the spent dash")
                self.assertEqual(start.to_dict(),original)

    def test_soft_and_original_route_no_dash_limits_cannot_be_bypassed_by_runup(self):
        for soft_limit,hard_limit in ((True,False),(False,True)):
            with self.subTest(soft=soft_limit,hard=hard_limit):
                sim,start,goal=runup_fixture()
                route=normalize_route(sim.map,dict(
                    waypoints=[dict(room=start.room,x=176,y=160)],
                    segments=[dict(no_dash=hard_limit,no_ground=False)]))
                wrapped=_ConstrainedSimulator(sim,route,no_dash=soft_limit,
                    phase=start_route_progress(route,start))
                stats={}
                self.assertEqual(refill_moves.refill_approach_edges(wrapped,start,goal,
                    deadline=time.monotonic()+2,max_expansions=256,stats=stats),[])
                self.assertLessEqual(stats["expansions"],256)

    def test_runup_respects_cumulative_budget_and_cannot_grant_partial_arrival(self):
        sim,start,goal=runup_fixture()
        stats={"expansions":251,"simulated_frames":0}
        edges=refill_moves.refill_approach_edges(sim,start,goal,
            deadline=time.monotonic()+2,max_expansions=256,stats=stats,max_frames=7)
        self.assertEqual(edges,[])
        self.assertGreater(stats["expansions"],251)
        self.assertLessEqual(stats["expansions"],256)
        self.assertLessEqual(stats["simulated_frames"],5*7)


if __name__=="__main__":unittest.main()
