"""Every discovered support can seed a continuation, including failed macros."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import normalize_route
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, JUMP, LEFT, RIGHT, Simulator
from celeste_search.planner import GroundArrival, Goal, ground_support_key, search_leg


def fixture(floor="111100011111000000000000000000", *, entities=()):
    room=Room("a",0,0,len(floor)*8,128,("0"*len(floor),)*12+(floor,)*4,
              tuple(entities),((16.,96.),))
    return MapData("arrivals",{"a":room},Path("arrivals.bin"))


def doomed_jump():
    # Land on the second support at frame 34, walk off at frame 57, die at 80.
    return tuple([RIGHT|JUMP]*16+[RIGHT]*140)


class PlannerGroundArrivalTests(unittest.TestCase):
    def search(self, data, macros, *, state=None, limit=4, expansions=1,
               simulator=None, goal=None, frames=300):
        sim=simulator or Simulator(data)
        start=state or sim.initial_state("a")
        arrivals=[]
        kwargs=dict(side_effect=macros) if callable(macros) else dict(return_value=macros)
        with patch("celeste_search.planner.macros",**kwargs):
            result=search_leg(sim,start,goal or Goal("target","a",4.,16.),
                deadline=time.monotonic()+5,max_expansions=expansions,max_frames=frames,
                ground_arrivals=arrivals,max_ground_arrivals_per_support=limit)
        counts=Counter(item.support_key for item in arrivals)
        self.assertTrue(all(count<=limit for count in counts.values()))
        for item in arrivals:
            self.assertIsInstance(item,GroundArrival)
            self.assertIsInstance(item.actions,tuple)
            current=start
            raw=Simulator(data)
            for action in item.actions:
                current=raw.step(current,action)
                self.assertFalse(current.dead)
            self.assertEqual(current,item.state)
            self.assertTrue(current.on_ground)
            self.assertGreaterEqual(current.vy,0.)
            self.assertEqual(ground_support_key(raw,current),item.support_key)
            self.assertLessEqual(len(item.actions),frames)
        return result,arrivals

    def test_macro_lands_then_leaves_and_dies_preserves_ground_prefix(self):
        result,arrivals=self.search(fixture(),[(doomed_jump(),len(doomed_jump()))])
        self.assertFalse(result.reached)
        self.assertEqual(result.actions,[])
        self.assertTrue(arrivals)
        self.assertTrue(all(34<=len(item.actions)<57 for item in arrivals))
        self.assertTrue(all(56<=item.state.x<100 for item in arrivals))

    def test_macro_ends_airborne_and_worse_than_goal_still_keeps_landing(self):
        sequence=doomed_jump()[:65]
        result,arrivals=self.search(fixture(),[(sequence,len(sequence))])
        self.assertFalse(result.reached)
        self.assertEqual(result.actions,[])
        self.assertTrue(any(item.state.x>60 for item in arrivals))

    def test_prefix_includes_all_ancestor_macros_and_original_frame(self):
        data=fixture("111100011111000011111000000000")
        start=replace(Simulator(data).initial_state("a"),frame=61,rem_x=.13,stamina=79.)
        def macros(state,*args):
            if state.x<32:
                sequence=doomed_jump()[:34]
                return [(sequence,len(sequence))]
            if state.x<80:
                return [(RIGHT,16)]
            sequence=doomed_jump()
            return [(sequence,len(sequence))]
        _,arrivals=self.search(data,macros,state=start,expansions=3)
        distant=[item for item in arrivals if item.state.x>=128]
        self.assertTrue(distant)
        self.assertTrue(all(len(item.actions)>=80 for item in distant))
        self.assertTrue(all(item.state.frame==61+len(item.actions) for item in arrivals))
        self.assertEqual(start.frame,61)
        self.assertEqual(start.rem_x,.13)
        self.assertEqual(start.stamina,79.)

    def test_every_support_retained_even_with_one_candidate_per_support(self):
        data=fixture()
        sequence=doomed_jump()
        _,arrivals=self.search(data,[(LEFT,12),(sequence,len(sequence))],limit=1)
        self.assertEqual(len({item.support_key for item in arrivals}),2)
        self.assertEqual(len(arrivals),2)
        self.assertTrue(any(item.state.x<=16 for item in arrivals))
        self.assertTrue(any(item.state.x>=56 for item in arrivals))

    def test_long_walk_is_bounded_per_support_and_keeps_reached_extremes(self):
        data=fixture("1"*100)
        _,arrivals=self.search(data,[(RIGHT,250)],limit=4)
        self.assertEqual(len(arrivals),4)
        self.assertEqual(len({item.support_key for item in arrivals}),1)
        self.assertLess(min(item.state.x for item in arrivals),20)
        self.assertGreater(max(item.state.x for item in arrivals),380)

    def test_arrival_is_collected_before_search_key_dominance_rejection(self):
        data=fixture("1"*30)
        with patch("celeste_search.planner.state_key",return_value=("same",)):
            result,arrivals=self.search(data,[(RIGHT,36)])
        self.assertFalse(result.reached)
        self.assertEqual(result.actions,[])
        self.assertTrue(any(item.state.x>60 for item in arrivals))

    def test_constraints_reject_landing_before_it_can_be_exported(self):
        data=fixture()
        sim=Simulator(data)
        route=normalize_route(data,dict(waypoints=[dict(room="a",x=176,y=88)],
                                      segments=[dict(no_ground=True)]))
        constrained=_ConstrainedSimulator(sim,route)
        _,arrivals=self.search(data,[(doomed_jump(),len(doomed_jump()))],simulator=constrained)
        self.assertEqual(arrivals,[])
        no_dash=normalize_route(data,dict(waypoints=[dict(room="a",x=176,y=88)],
                                        segments=[dict(no_dash=True)]))
        _,arrivals=self.search(data,[(RIGHT|DASH,20)],
            simulator=_ConstrainedSimulator(Simulator(data),no_dash))
        self.assertEqual(arrivals,[])

    def test_moving_support_identity_is_stable_and_relative_position_matters(self):
        entity=Entity("fallingBlock",7,64,48,32,16)
        data=fixture(entities=(entity,))
        sim=Simulator(data)
        first=replace(sim.initial_state("a"),x=80.,y=48.,on_ground=True)
        second=replace(first,x=144.,y=72.,mechanisms=(("a:7",2,6,128.,72.,80.),))
        key=ground_support_key(sim,first)
        self.assertEqual(key,("mechanism","a","a:7"))
        self.assertEqual(ground_support_key(sim,second),key)
        vanished=replace(second,mechanisms=(("a:7",4,0,128.,144.,0.),))
        self.assertIsNone(ground_support_key(sim,vanished))

    def test_ground_flag_without_actual_support_is_not_a_candidate(self):
        data=fixture()
        sim=Simulator(data)
        start=sim.initial_state("a")
        self.assertIsNone(ground_support_key(sim,replace(start,x=44.,on_ground=True)))
        self.assertIsNone(ground_support_key(sim,replace(start,y=80.,on_ground=True)))
        self.assertIsNone(ground_support_key(sim,replace(start,vy=-10.)))
        self.assertIsNone(ground_support_key(sim,replace(start,dead=True)))

    def test_permanent_obstruction_splits_support_and_exact_wall_contact_counts(self):
        data=fixture("1"*30,entities=(Entity("solid",7,80,64,8,32),))
        sim=Simulator(data)
        start=sim.initial_state("a")
        left=ground_support_key(sim,replace(start,x=76.))
        right=ground_support_key(sim,replace(start,x=92.))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertNotEqual(left,right)
        self.assertEqual(left,ground_support_key(sim,replace(start,x=32.)))

    def test_spike_strip_separates_safe_islands_on_one_continuous_floor(self):
        data=fixture("1"*30,entities=(Entity("spikesUp",8,80,96,16,3),))
        sim=Simulator(data)
        start=sim.initial_state("a")
        left=ground_support_key(sim,replace(start,x=64.))
        right=ground_support_key(sim,replace(start,x=112.))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertNotEqual(left,right)
        self.assertEqual(left,ground_support_key(sim,replace(start,x=32.)))
        self.assertEqual(right,ground_support_key(sim,replace(start,x=200.)))

    def test_attached_spike_islands_follow_actual_pose_and_stable_identity(self):
        entities=(Entity("fallingBlock",7,64,48,48,16),
                  Entity("spikesUp",8,80,48,8,3))
        data=fixture(entities=entities)
        sim=Simulator(data)
        start=replace(sim.initial_state("a"),x=72.,y=48.)
        left=ground_support_key(sim,start)
        self.assertNotEqual(left,ground_support_key(sim,replace(start,x=104.)))
        moved=replace(start,x=136.,y=72.,mechanisms=(("a:7",2,6,128.,72.,80.),))
        self.assertEqual(left,ground_support_key(sim,moved))
        self.assertNotEqual(left,ground_support_key(sim,replace(moved,x=168.)))

    def test_dynamic_obstacle_state_and_clock_are_preserved_in_every_prefix(self):
        data=fixture("1"*30,entities=(Entity("zipMover",7,64,64,24,16,{},((128.,64.),)),))
        start=replace(Simulator(data).initial_state("a"),frame=42,
                      mechanisms=(("a:7",3,7,128.,64.,0.),))
        _,arrivals=self.search(data,[(RIGHT,80)],state=start)
        self.assertTrue(arrivals)
        self.assertTrue(any(item.state.mechanisms!=start.mechanisms for item in arrivals))
        self.assertTrue(all(item.state.frame==42+len(item.actions) for item in arrivals))

    def test_enabling_collection_does_not_change_the_search_result(self):
        data=fixture("1"*30)
        goal=Goal("target","a",200.,32.)
        results=[]
        for enabled in (False,True):
            sim=Simulator(data)
            kwargs=dict(ground_arrivals=[]) if enabled else {}
            results.append(search_leg(sim,sim.initial_state(),goal,
                deadline=time.monotonic()+5,max_expansions=30,max_frames=150,**kwargs))
        self.assertEqual(results[0],results[1])

    def test_unsupported_room_never_exports_a_continuation_seed(self):
        data=fixture("1"*30,entities=(Entity("unmodeledTrap",9,200,48,16,16),))
        result,arrivals=self.search(data,[(RIGHT,36)])
        self.assertFalse(result.reached)
        self.assertEqual(arrivals,[])

    def test_nonpositive_limit_is_rejected_only_when_collection_is_enabled(self):
        sim=Simulator(fixture())
        kwargs=dict(deadline=time.monotonic()+1,max_expansions=0,
                    max_ground_arrivals_per_support=0)
        with self.assertRaises(ValueError):
            search_leg(sim,sim.initial_state(),Goal("target","a",200,0),ground_arrivals=[],**kwargs)
        search_leg(sim,sim.initial_state(),Goal("target","a",200,0),**kwargs)


if __name__=="__main__":
    unittest.main()
