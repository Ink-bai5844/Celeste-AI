"""Ground regions are physically connected private guidance, never success."""
from dataclasses import replace
from pathlib import Path
import json
import time
import unittest

from celeste_search.assist_planning import _selected_goal, target_reached
from celeste_search.assist_routes import (advance_route_progress, normalize_route,
                                         route_complete, start_route_progress)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DOWN, GRAB, JUMP, RIGHT, UP, Simulator
from celeste_search.regional_planning import (GroundInterval, LandingRegionGoal,
    _WalkJumpSimulator, ground_intervals, search_regional_routes, target_ground_intervals)
from celeste_search.room_planning import action_fingerprint


def flat_fixture(width=192, *, entities=()):
    columns=width//8
    room=Room("a",0,0,width,112,("0"*columns,)*12+("1"*columns,)*2,
              tuple(entities),((16.,96.),))
    return MapData("regional",{"a":room},Path("regional.bin"))


def gap_fixture(mirror=False):
    rows=[["0"]*40 for _ in range(20)]
    for y in range(16,20):
        for x in range(8): rows[y][x]="1"
    for y in range(12,20):
        for x in range(18,38): rows[y][x]="1"
    if mirror:
        rows=[list(reversed(row)) for row in rows]
    room=Room("a",0,0,320,160,tuple("".join(row) for row in rows),(),
              ((296. if mirror else 24.,128.),))
    return MapData("regional_gap",{"a":room},Path("regional_gap.bin"))


class RegionalPlanningTests(unittest.TestCase):
    def run_search(self, data, points, *, constraints=None, state=None,
                   frames=600, expansions=2000, forbidden=()):
        sim=Simulator(data)
        start=state or sim.initial_state("a")
        route=normalize_route(data,dict(waypoints=points,
            segments=constraints or [{} for _ in points]))
        goals=[_selected_goal(point)[0] for point in route["waypoints"]]
        result=search_regional_routes(sim,start,route,goals,
            deadline=time.monotonic()+8,max_expansions=expansions,
            max_frames=frames,forbidden=forbidden)
        # Verification uses a fresh uncached simulator and the *exact* original
        # snapshot, including frame, inventory, timers and mechanism records.
        raw=Simulator(data)
        actual=start
        progress=start_route_progress(route,start)
        for mask in result[0]:
            before=actual
            actual=raw.step(actual,mask)
            progress=advance_route_progress(route,progress,before,actual,mask)
            self.assertFalse(actual.dead)
            self.assertIsNone(progress.violation)
        self.assertEqual(actual,result[1])
        self.assertEqual(progress,result[5])
        self.assertEqual(route_complete(route,progress),result[2])
        if result[2]:
            self.assertTrue(target_reached(actual,route["waypoints"][-1]))
        self.assertLessEqual(len(result[0]),frames)
        self.assertLessEqual(result[3],expansions)
        json.dumps(result[6])
        return result,route,start

    def test_flat_spans_are_continuous_and_obstacles_split_clearance(self):
        data=flat_fixture(entities=(Entity("solid",2,80,64,8,32),
                                    Entity("spikesUp",3,120,96,16,3)))
        sim=Simulator(data)
        intervals=ground_intervals(sim,sim.initial_state("a"))
        floor=[r for r in intervals if r.y==96]
        self.assertTrue(any(r.left==4 and r.right==76 for r in floor))
        self.assertFalse(any(r.left<84<r.right for r in floor))
        self.assertFalse(any(r.left<=128<=r.right for r in floor))
        self.assertTrue(any(r.right==188 for r in floor))

    def test_actual_walk_prefix_and_jump_from_long_platform(self):
        data=flat_fixture(800)
        result,_,start=self.run_search(data,[dict(room="a",x=720,y=64)],frames=800)
        self.assertTrue(result[2],result[6])
        self.assertGreater(len(result[0]),320)
        self.assertIn("ground_start_walk",result[6]["stages"])
        self.assertIn("ground_start_jump",result[6]["stages"])
        self.assertFalse(any(mask&DASH for mask in result[0]))
        self.assertEqual(start.x,16.)
        self.assertEqual(start.frame,0)

    def test_enters_far_end_of_target_floor_then_hits_exact_tile(self):
        data=gap_fixture()
        result,route,_=self.run_search(data,[dict(room="a",x=272,y=88)])
        self.assertTrue(result[2],result[6])
        self.assertTrue({"ground_region_landing", "ground_frontier_landing"}
                        .intersection(result[6]["stages"]))
        self.assertTrue({"ground_exact_suffix", "ground_start_walk"}
                        .intersection(result[6]["stages"]))
        self.assertTrue(all(not row["reached"] for row in result[6]["attempts"]
                            if row["kind"]=="ground_exact_no_dash"))
        # A failed exact probe can now discover the same floor before a
        # separate target-region proposal is necessary.
        sim=Simulator(data)
        goal=_selected_goal(route["waypoints"][0])[0]
        self.assertTrue(any(row.left==148 and row.right==300
                            for row in target_ground_intervals(sim,sim.initial_state("a"),goal)))
        self.assertTrue(target_reached(result[1],route["waypoints"][0]))

    def test_mirrored_region_connection_keeps_exact_target(self):
        result,_,_=self.run_search(gap_fixture(True),[dict(room="a",x=40,y=88)])
        self.assertTrue(result[2],result[6])
        self.assertFalse(result[1].dead)
        self.assertLess(result[1].x,48)

    def test_ordered_route_keeps_original_momentum_and_segment_limits(self):
        data=flat_fixture()
        state=replace(Simulator(data).initial_state("a"),frame=73,stamina=53.,
                      rem_x=.2,rem_y=.1)
        result,_,_=self.run_search(data,[dict(room="a",x=160,y=88),
            dict(room="a",x=32,y=88),dict(room="a",x=128,y=72)],
            state=state,constraints=[dict(no_dash=True)]*3)
        self.assertTrue(result[2],result[6])
        self.assertEqual(result[5].index,3)
        self.assertEqual(result[1].frame,73+len(result[0]))
        self.assertFalse(any(mask&DASH for mask in result[0]))
        self.assertEqual(state.stamina,53.)
        self.assertEqual(state.rem_x,.2)

    def test_no_ground_cannot_complete_by_landing_on_goal_floor(self):
        result,_,_=self.run_search(gap_fixture(),[dict(room="a",x=272,y=88)],
            constraints=[dict(no_ground=True,no_dash=True)],expansions=600)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,0)
        self.assertEqual(result[6]["target_regions"],[])

    def test_private_floor_is_not_success_when_exact_tile_is_unreachable(self):
        # The selected empty tile is sealed in a low ceiling cavity. Nearby
        # exposed floor is valid guidance, but the player cannot fit inside it.
        rows=[list(row) for row in flat_fixture().rooms["a"].solids]
        for y in range(4,8):
            for x in range(14,23): rows[y][x]="1"
        rows[6][18]="0"
        room=replace(flat_fixture().rooms["a"],solids=tuple("".join(row) for row in rows))
        data=MapData("sealed",{"a":room},Path("sealed.bin"))
        result,_,_=self.run_search(data,[dict(room="a",x=144,y=48)],expansions=500)
        self.assertFalse(result[2])
        self.assertEqual(result[5].index,0)
        self.assertEqual(result[1].x,16.)

    def test_dynamic_support_goal_follows_motion_and_disappearance(self):
        entity=Entity("fallingBlock",7,64,48,24,16)
        data=flat_fixture(entities=(entity,))
        sim=Simulator(data)
        state=sim.initial_state("a")
        region=next(r for r in ground_intervals(sim,state) if r.identity=="a:7")
        goal=LandingRegionGoal("surface","a",76,48,region=region,simulator=sim)
        shifted=replace(state,x=84.,y=72.,on_ground=True,vy=0.,
                        mechanisms=(("a:7",2,6,72.,72.,80.),))
        self.assertEqual(goal.waypoint(shifted),(84.,72.))
        self.assertTrue(goal.achieved(shifted))
        self.assertFalse(goal.achieved(replace(shifted,x=76.,y=48.)))
        gone=replace(shifted,mechanisms=(("a:7",4,0,72.,128.,0.),))
        self.assertFalse(goal.achieved(gone))
        self.assertFalse(any(r.identity=="a:7" for r in ground_intervals(sim,gone)))

    def test_moving_solid_clearance_uses_current_snapshot(self):
        data=flat_fixture(entities=(Entity("zipMover",7,64,64,24,32,{},((128.,64.),)),))
        sim=Simulator(data)
        start=sim.initial_state("a")
        def contains(state,x):
            return any(r.y==96 and r.left<=x<=r.right for r in ground_intervals(sim,state))
        self.assertFalse(contains(start,76))
        self.assertTrue(contains(start,140))
        moved=replace(start,mechanisms=(("a:7",3,0,128.,64.,0.),))
        self.assertTrue(contains(moved,76))
        self.assertFalse(contains(moved,140))

    def test_moving_obstacle_clock_survives_whole_physical_connection(self):
        data=flat_fixture(entities=(Entity("zipMover",7,64,64,24,16,{},((128.,64.),)),))
        state=replace(Simulator(data).initial_state("a"),frame=29,
                      mechanisms=(("a:7",3,7,128.,64.,0.),))
        result,_,_=self.run_search(data,[dict(room="a",x=160,y=88)],state=state)
        self.assertTrue(result[2],result[6])
        self.assertEqual(result[1].frame,29+len(result[0]))
        self.assertNotEqual(result[1].mechanisms,state.mechanisms)
        self.assertEqual(state.mechanisms,(("a:7",3,7,128.,64.,0.),))

    def test_ordinary_suffix_disallows_climbing_and_dash_shortcuts(self):
        data=flat_fixture()
        sim=Simulator(data)
        start=sim.initial_state("a")
        route=normalize_route(data,dict(room="a",x=160,y=88))
        ordinary=_WalkJumpSimulator(sim,route,no_dash=True)
        for mask in (DASH,GRAB,UP,DOWN,RIGHT|GRAB,JUMP|DASH):
            result=ordinary.step(start,mask)
            self.assertTrue(result.dead)
            self.assertEqual(result.reason,"regional_walk_jump_only")
            self.assertEqual(result.frame,start.frame)
        self.assertFalse(ordinary.step(start,RIGHT|JUMP).dead)

    def test_forbidden_complete_candidate_tries_another_real_prefix(self):
        data=flat_fixture()
        points=[dict(room="a",x=96,y=88)]
        first,_,_=self.run_search(data,points)
        result,_,_=self.run_search(data,points,
            forbidden=[action_fingerprint(first[0])],expansions=800)
        self.assertTrue(result[2],result[6])
        self.assertNotEqual(action_fingerprint(first[0]),action_fingerprint(result[0]))
        self.assertGreater(result[6]["excluded_candidates"],0)

    def test_no_budget_never_fabricates_start_or_success(self):
        result,_,start=self.run_search(flat_fixture(),[dict(room="a",x=96,y=88)],expansions=0)
        self.assertEqual(result[0],[])
        self.assertEqual(result[1],start)
        self.assertFalse(result[2])


if __name__=="__main__":
    unittest.main()
