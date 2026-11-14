"""Automatic playback retries preserve the complete request and human priority."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from celeste_search.assist import AssistController, AssistOptions, PASSIVE
from celeste_search.assist_routes import advance_route_progress, route_complete, start_route_progress
from celeste_search.maps import MapData, Room
from celeste_search.physics import RIGHT, Simulator
from celeste_search.room_planning import action_fingerprint
from celeste_search.routes import snapshot
from test_assist_controller import FakeTargetPlanner, observation


class AutomaticRetryTests(unittest.TestCase):
    def setUp(self):
        room=Room('a',0,0,128,80,('0'*16,)*8+('1'*16,)*2,(),((24.,64.),))
        self.map=MapData('automatic-retries',{'a':room},Path('fixture.bin'))
        self.sim=Simulator(self.map)
        self.start=self.sim.initial_state('a')
        self.planner=FakeTargetPlanner(self.sim)
        self.controller=AssistController(self.map,self.planner)
        self.route=dict(waypoints=[dict(room='a',x=56,y=56,tile_size=8),
                                  dict(room='a',x=96,y=56,tile_size=8)],
                        segments=[dict(no_dash=False,no_ground=False) for _ in range(2)])
        self.frame=0

    def tick(self,state=None,**changes):
        self.frame+=1
        return self.controller.observe(observation(state or self.start,self.frame,**changes))

    def prime(self):
        self.controller.confirm_route(self.route)
        for _ in range(3): self.assertEqual(self.tick(),PASSIVE)
        self.assertEqual(self.controller.mode,'planning')

    def candidate(self,delay=0):
        state=self.planner.requests[-1]['state']
        phase=start_route_progress(self.route,state)
        actions=[];states=[state];indices=[]
        for mask in [0]*delay+[RIGHT]*180:
            before,old=state,phase.index
            state=self.sim.step(state,mask)
            phase=advance_route_progress(self.route,phase,before,state,mask)
            actions.append(mask);states.append(state)
            indices.extend([len(actions)]*(phase.index-old))
            if route_complete(self.route,phase):break
        self.assertTrue(route_complete(self.route,phase))
        return dict(can_execute=True,reached=True,route=deepcopy(self.route),actions=actions,
            expected=[snapshot(s) for s in states],anchor_state=states[0].to_dict(),
            final_state=state.to_dict(),waypoint_indices=indices,
            fingerprint=action_fingerprint(actions)),states

    def activate(self):
        self.prime();candidate,states=self.candidate()
        self.planner.publish(candidate)
        self.assertEqual(self.tick(),RIGHT)
        return candidate,states

    def test_dead_prefix_is_excluded_and_next_complete_candidate_can_succeed(self):
        candidate,states=self.activate()
        first=candidate['waypoint_indices'][0]
        for state in states[1:first+1]:self.tick(state)
        self.assertEqual(self.controller.progress.index,1)
        self.assertEqual(self.tick(states[first],dead=True),PASSIVE)
        self.assertIsNone(self.controller.progress)
        self.assertEqual(self.controller.index,0)
        for _ in range(4):self.tick(spawn_id=2,deaths=1)
        self.assertEqual(len(self.planner.requests),2)
        request=self.planner.requests[-1]
        self.assertEqual(request['target'],self.route)
        self.assertEqual(request['forbidden'],{candidate['fingerprint']})
        replacement,new_states=self.candidate(delay=1)
        self.assertNotEqual(replacement['fingerprint'],candidate['fingerprint'])
        self.planner.publish(replacement)
        self.assertEqual(self.tick(spawn_id=2,deaths=1),0)
        for state in new_states[1:]:self.tick(state,spawn_id=2,deaths=1)
        self.assertEqual(self.controller.mode,'success')
        self.assertEqual(self.controller.progress.index,2)

    def test_duplicate_failed_worker_candidate_never_takes_control(self):
        candidate,_=self.activate()
        self.tick(dead=True)
        for _ in range(4):self.tick(spawn_id=2,deaths=1)
        duplicate,_=self.candidate()
        self.assertEqual(duplicate['fingerprint'],candidate['fingerprint'])
        self.planner.publish(duplicate)
        self.assertEqual(self.tick(spawn_id=2,deaths=1),PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertEqual(self.controller.mode,'waiting')
        self.assertTrue(any(e['event']=='failed_candidate_excluded' for e in self.controller.events))

    def test_action_tail_waits_then_replans_without_awarding_unvisited_points(self):
        self.prime();candidate,_=self.candidate()
        candidate['actions']=candidate['actions'][:1]
        candidate['expected']=candidate['expected'][:2]
        candidate['waypoint_indices']=[1,1]
        self.planner.publish(candidate);self.tick()
        self.assertEqual(self.tick(),PASSIVE)
        self.assertTrue(self.controller.recovery['allow_completion'])
        self.assertNotEqual(self.controller.result,'reached')
        for _ in range(4):self.tick()
        self.assertEqual(len(self.planner.requests),2)
        self.assertIn(action_fingerprint([RIGHT]),self.planner.requests[-1]['forbidden'])
        self.assertEqual(self.controller.status()['waypoint_index'],0)

    def test_eight_failed_search_rounds_end_without_claiming_impossibility(self):
        self.prime()
        for attempt in range(8):
            self.assertEqual(len(self.planner.requests),attempt+1)
            self.planner.publish(dict(can_execute=False,reason='budget_exhausted'))
            self.assertEqual(self.tick(),PASSIVE)
            if attempt<7:
                for _ in range(3):self.tick()
        self.assertEqual(self.controller.mode,'failed')
        self.assertIn('不代表目标不可达',self.controller.message)
        for _ in range(10):self.tick()
        self.assertEqual(len(self.planner.requests),8)

    def test_human_input_cancels_planning_and_respawn_wait_without_restart(self):
        for recovering in (False,True):
            with self.subTest(recovering=recovering):
                self.setUp()
                if recovering:self.activate();self.tick(dead=True)
                else:self.prime()
                self.assertEqual(self.tick(human_input_mask=RIGHT,human_input_epoch=4),PASSIVE)
                self.assertEqual(self.controller.mode,'manual')
                for _ in range(10):self.tick(spawn_id=2,deaths=1,human_input_epoch=4)
                self.assertEqual(len(self.planner.requests),1)

    def test_execution_and_recovery_do_not_spend_next_round_search_budget(self):
        with patch('celeste_search.assist.time.monotonic',return_value=100):
            self.activate()
        with patch('celeste_search.assist.time.monotonic',return_value=120):
            self.tick(dead=True)
        with patch('celeste_search.assist.time.monotonic',return_value=130):
            for _ in range(4):self.tick(spawn_id=2,deaths=1)
        self.assertEqual(self.controller.started,130)
        self.assertEqual(self.controller.options.max_search_seconds,120)
        self.assertEqual(self.controller.mode,'planning')
        with patch('celeste_search.assist.time.monotonic',return_value=219):
            self.tick(spawn_id=2,deaths=1)
        self.assertEqual(len(self.planner.requests),2)
        self.assertEqual(self.controller.mode,'planning')

    def test_historical_native_request7_drift_continues_then_respawn_discards_route(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/assist-request7-drift.json').read_text())
        actual=fixture['observation'];expected=fixture['expected']
        room=Room('02b',736,-560,400,192,('0'*50,)*24,(),())
        destination=replace(room,name='03',y=-752)
        data=MapData('historical-native-controller',{'02b':room,'03':destination},Path('fixture.bin'))
        controller=AssistController(data,SimpleNamespace(cancel=lambda:None))
        controller.route_request=fixture['route'];controller.target=fixture['route']['waypoints'][-1]
        controller.mode='executing';controller.started=controller.executing_started=time.monotonic()
        controller.execution_spawn=actual['spawn_id'];controller.execution_deaths=actual['deaths']
        controller.request_human_epoch=actual['human_input_epoch']
        controller.progress=start_route_progress(controller.route_request,actual)
        controller.execution_previous=deepcopy(actual)
        controller.plan=dict(actions=fixture['next_actions'],expected=[expected]*4)
        self.assertEqual(abs(expected['y']-actual['y']),11)
        self.assertEqual(controller.observe(actual),fixture['next_actions'][0])
        self.assertEqual(controller.mode,'executing')
        self.assertEqual(controller.execution_warnings[0]['kind'],'player_position')
        self.assertFalse(controller.forbidden)
        self.assertEqual(controller.observe(fixture['respawn']),PASSIVE)
        self.assertIsNone(controller.progress)
        self.assertEqual(controller.index,0)
        self.assertEqual(controller.forbidden,{action_fingerprint(fixture['next_actions'])})
        self.assertEqual(controller.mode,'waiting')


if __name__=='__main__':unittest.main()
