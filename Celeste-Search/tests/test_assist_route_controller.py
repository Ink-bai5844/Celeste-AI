"""Whole-itinerary takeover and failure checks, without launching the game."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist import AssistController, PASSIVE
from celeste_search.assist_routes import (start_route_progress, advance_route_progress,
                                         route_complete)
from celeste_search.maps import MapData, Room
from celeste_search.physics import DASH, RIGHT, Simulator
from celeste_search.room_planning import action_fingerprint
from celeste_search.routes import snapshot
from test_assist_controller import FakeTargetPlanner, observation


class RouteControllerTests(unittest.TestCase):
    def setUp(self):
        rows=("0"*16,)*8+("1"*16,)*2
        room=Room("a",0,0,128,80,rows,(),((24.,64.),))
        other=replace(room,name="b",x=128,spawns=((152.,64.),))
        self.data=MapData("route-controller",{"a":room,"b":other},Path("fixture.bin"))
        self.sim=Simulator(self.data)
        self.start=self.sim.initial_state("a")
        self.planner=FakeTargetPlanner(self.sim)
        self.controller=AssistController(self.data,self.planner)
        self.route={"waypoints":[{"room":"a","x":56,"y":56,"tile_size":8},
                                 {"room":"a","x":96,"y":56,"tile_size":8}],
                    "segments":[{"no_dash":False,"no_ground":False} for _ in range(2)]}

    def prime(self):
        self.controller.confirm_route(self.route)
        for frame in range(1,4):
            self.assertEqual(self.controller.observe(observation(self.start,frame)),PASSIVE)
        self.assertEqual(self.controller.mode,"planning")
        self.assertEqual(self.planner.requests[0]["target"],self.route)

    def candidate(self):
        state=self.planner.requests[-1]["state"]
        progress=start_route_progress(self.route,state)
        states=[state]
        actions=[]
        indices=[0]*progress.index
        for _ in range(180):
            previous=state
            state=self.sim.step(state,RIGHT)
            old_progress=progress
            progress=advance_route_progress(self.route,progress,previous,state,RIGHT)
            states.append(state)
            actions.append(RIGHT)
            indices.extend([len(actions)]*(progress.index-old_progress.index))
            if route_complete(self.route,progress):
                break
        self.assertTrue(route_complete(self.route,progress))
        return dict(can_execute=True,reached=True,route=deepcopy(self.route),
                    actions=actions,expected=[snapshot(s) for s in states],
                    anchor_state=states[0].to_dict(),final_state=states[-1].to_dict(),
                    waypoint_indices=indices,fingerprint=action_fingerprint(actions)),states

    def activate(self):
        self.prime()
        candidate,states=self.candidate()
        self.planner.publish(candidate)
        self.assertEqual(self.controller.observe(observation(self.start,4)),RIGHT)
        return candidate,states

    def test_all_waypoints_are_one_search_and_one_uninterrupted_execution(self):
        candidate,states=self.activate()
        for index,state in enumerate(states[1:],1):
            action=self.controller.observe(observation(state,index+4,control_mode="assist"))
            if index==candidate["waypoint_indices"][0]:
                self.assertEqual(self.controller.status()["waypoint_index"],1)
                self.assertEqual(self.controller.mode,"executing")
                self.assertEqual(action,RIGHT)
            if index<len(candidate["actions"]):
                self.assertEqual(action,RIGHT)
        self.assertEqual(self.controller.mode,"success")
        self.assertEqual(action,PASSIVE)
        self.assertEqual(len(self.planner.requests),1)

    def test_planned_room_transition_keeps_the_whole_tail(self):
        self.route["waypoints"][1].update(room="b",x=144)
        candidate,states=self.activate()
        self.assertIn("b",{state.room for state in states})
        for index,state in enumerate(states[1:],1):
            self.controller.observe(observation(state,index+4,control_mode="assist"))
        self.assertEqual(self.controller.mode,"success")
        self.assertEqual(len(self.planner.requests),1)

    def test_death_discards_completed_prefix_and_replans_entire_route(self):
        candidate,states=self.activate()
        for index,state in enumerate(states[1:candidate["waypoint_indices"][0]+1],1):
            self.controller.observe(observation(state,index+4,control_mode="assist"))
        self.assertEqual(self.controller.status()["waypoint_index"],1)
        answer=self.controller.observe(observation(states[-1],100,dead=True,control_mode="assist"))
        self.assertEqual(answer,PASSIVE)
        self.assertEqual(self.controller.mode,"waiting")
        self.assertEqual(self.controller.status()["waypoint_index"],0)
        self.assertEqual(self.controller.index,0)
        for frame in range(101,110):
            self.assertEqual(self.controller.observe(observation(self.start,frame,spawn_id=2,deaths=1)),PASSIVE)
        self.assertEqual(len(self.planner.requests),2)
        self.assertEqual(self.planner.requests[-1]["target"],self.route)
        self.assertIn(candidate["fingerprint"],self.planner.requests[-1]["forbidden"])

    def test_death_counter_change_at_final_point_cannot_award_success(self):
        _,states=self.activate()
        self.assertEqual(self.controller.observe(observation(states[-1],100,spawn_id=2,deaths=1)),PASSIVE)
        self.assertEqual(self.controller.mode,"waiting")
        self.assertIsNone(self.controller.progress)

    def test_final_target_first_does_not_skip_intermediate_point(self):
        _,states=self.activate()
        self.controller.observe(observation(states[-1],100,control_mode="assist"))
        self.assertEqual(self.controller.mode,"executing")
        self.assertEqual(self.controller.progress.index,0)
        self.assertNotEqual(self.controller.result,"reached")

    def test_confirmed_route_is_a_snapshot_of_draft(self):
        self.prime()
        self.route["waypoints"][0]["x"]=80
        self.route["segments"][1]["no_dash"]=True
        self.assertEqual(self.controller.route_request["waypoints"][0]["x"],56)
        self.assertFalse(self.controller.route_request["segments"][1]["no_dash"])

    def test_wrong_route_or_prefix_only_candidate_cannot_take_control(self):
        for problem in ("route","indices","incomplete"):
            with self.subTest(problem=problem):
                self.setUp()
                self.prime()
                candidate,_=self.candidate()
                if problem=="route":
                    candidate["route"]["segments"][0]["no_dash"]=True
                elif problem=="indices":
                    candidate["waypoint_indices"].pop()
                else:
                    candidate["reached"]=False
                self.planner.publish(candidate)
                self.assertEqual(self.controller.observe(observation(self.start,4)),PASSIVE)
                self.assertEqual(self.controller.mode,"failed")

    def test_no_dash_never_dispatches_a_prohibited_first_mask(self):
        self.route["segments"][0]["no_dash"]=True
        self.prime()
        candidate,_=self.candidate()
        candidate["actions"][0]|=DASH
        self.planner.publish(candidate)
        self.assertEqual(self.controller.observe(observation(self.start,4)),PASSIVE)
        self.assertEqual(self.controller.mode,"waiting")
        self.assertIsNotNone(self.controller.recovery)

    def test_actual_unpredicted_dash_is_detected_before_next_input(self):
        self.route["segments"][0]["no_dash"]=True
        _,states=self.activate()
        dashed=replace(states[1],state=2,dash_frames=5)
        self.assertEqual(self.controller.observe(observation(dashed,5,control_mode="assist")),PASSIVE)
        self.assertEqual(self.controller.mode,"waiting")
        self.assertIsNotNone(self.controller.recovery)

    def test_airborne_segment_landing_fails_even_at_its_target(self):
        candidate,states=self.activate()
        # Directly exercise the execution guard with a physically possible air
        # segment start; actual planner/replay tests validate candidate search.
        self.controller.route_request["segments"][0]["no_ground"]=True
        before=replace(states[1],on_ground=False,y=58)
        self.controller.progress=start_route_progress(self.controller.route_request,before)
        self.controller.execution_previous=observation(before,5)
        landed=replace(states[1],on_ground=True,x=56)
        self.assertEqual(self.controller.observe(observation(landed,6,control_mode="assist")),PASSIVE)
        self.assertEqual(self.controller.mode,"waiting")
        self.assertIsNotNone(self.controller.recovery)

    def test_cancel_releases_inputs_and_preserves_no_automatic_restart(self):
        self.activate()
        self.controller.cancel()
        self.assertEqual(self.controller.observe(observation(self.start,5)),PASSIVE)
        self.assertEqual(self.controller.mode,"manual")
        self.assertEqual(len(self.planner.requests),1)


if __name__=="__main__":
    unittest.main()
