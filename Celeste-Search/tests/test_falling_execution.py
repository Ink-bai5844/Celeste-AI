"""Execution records falling-state drift without abandoning validated input playback.

These tests use the real native FallingBlock decoder and controller, but no
game, sockets, worker pool, or wall-clock search.
"""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from celeste_search.assist import AssistController, PASSIVE
from celeste_search.assist_routes import start_route_progress
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DT, RIGHT, SimState
from celeste_search.routes import snapshot
from celeste_search.runtime_state import falling_snapshot_from_observation


def native_block(record):
    identity,phase,timer,exact_x,exact_y,speed=record
    x,y=math.floor(exact_x),math.floor(exact_y)
    native=1 if phase==0 else 3 if phase==1 else 8
    return dict(id=identity,room='a',entity_id=7,kind='fallingBlock',
        present=True,active=True,collidable=True,retained_observation=False,
        x=x,y=y,width=16.,height=16.,rem_x=exact_x-x,rem_y=exact_y-y,
        lift_vx=0.,lift_vy=speed,triggered=False,fall_delay=0.,
        has_started_falling=phase!=0,unavailable=['coroutine_phase'],
        coroutines=[dict(active=True,finished=False,
            wait_timer=(12-timer)*DT if phase==1 else 0.,
            stack=[dict(iterator_type='Celeste.FallingBlock+<Sequence>d__21',
                fields={'<>1__state':native,'<speed>5__2':speed,
                        '<maxSpeed>5__3':160. if phase==2 else 0.,'<timer>5__4':0.})])])


class FallingExecutionTests(unittest.TestCase):
    def setup_execution(self, *, record=('a:7',0,0,80.,16.,0.), falling=True):
        entities=(Entity('fallingBlock',7,80.,16.,16.,16.,dict(climbFall=True)),) if falling else ()
        room=Room('a',0,0,160,112,('0'*20,)*12+('1'*20,)*2,entities,((24.,96.),))
        data=MapData('falling-execution',{'a':room},Path('fixture.bin'))
        state=SimState('a',24.,96.,on_ground=True,frame=100,
                       mechanisms=(record,) if falling else ())
        observation=state.to_dict()
        observation.update(model_state=state.to_dict(),model_state_unavailable=[],
            human_input_mask=0,human_menu_input=False,human_input_epoch=3,
            spawn_id=1,deaths=0,control_mode='manual',input_bindings_supported=True,
            input_phase='before_virtual_inputs',on_safe_ground=True,
            delta_time=DT,raw_delta_time=DT,area_id=1,area_mode=0,
            room_entities=dict(entities=[native_block(record)] if falling else [],
                               do_not_load=[],unavailable=[]))
        route=dict(waypoints=[dict(room='a',x=136,y=88,tile_size=8)],
                   segments=[dict(no_dash=False,no_ground=False)])
        controller=AssistController(data,SimpleNamespace(cancel=lambda:None))
        controller.mode='executing'
        controller.target=route['waypoints'][0]
        controller.route_request=route
        controller.progress=start_route_progress(route,state)
        controller.execution_previous=deepcopy(observation)
        controller.execution_spawn=1
        controller.execution_deaths=0
        controller.execution_human_epoch=3
        controller.started=controller.executing_started=time.monotonic()
        controller.plan=dict(actions=[RIGHT,RIGHT],expected=[snapshot(state)]*3,
                             anchor_state=state.to_dict(),fingerprint='fixture')
        return controller,observation,state

    def assertContinuesWithWarning(self,controller,observation):
        position=(observation['room'],observation['x'],observation['y'])
        predicted=controller.plan['expected'][0]
        self.assertEqual(position,(predicted['room'],predicted['x'],predicted['y']))
        self.assertEqual(controller.observe(observation),RIGHT)
        self.assertEqual(controller.mode,'executing')
        self.assertEqual(controller.index,1)
        self.assertTrue(controller.execution_warnings)

    def test_matching_complete_native_mechanism_allows_next_input(self):
        controller,observation,state=self.setup_execution()
        self.assertEqual(falling_snapshot_from_observation(controller.map,observation),
                         [list(state.mechanisms[0])])
        self.assertEqual(controller.observe(observation),RIGHT)
        self.assertEqual(controller.mode,'executing')
        self.assertEqual(controller.index,1)

    def test_phase_drift_warns_before_player_moves(self):
        controller,observation,_=self.setup_execution()
        observation['room_entities']['entities'][0]=native_block(('a:7',1,6,80.,16.,0.))
        self.assertContinuesWithWarning(controller,observation)

    def test_timer_drift_warns_before_player_moves(self):
        controller,observation,_=self.setup_execution(record=('a:7',1,6,80.,16.,0.))
        observation['room_entities']['entities'][0]=native_block(('a:7',1,8,80.,16.,0.))
        self.assertContinuesWithWarning(controller,observation)

    def test_exact_position_and_speed_drift_warns_before_player_moves(self):
        expected=('a:7',2,0,80.,40.25,30.)
        for actual in [('a:7',2,0,80.,41.25,30.),('a:7',2,0,80.,40.25,38.)]:
            with self.subTest(actual=actual):
                controller,observation,_=self.setup_execution(record=expected)
                observation['room_entities']['entities'][0]=native_block(actual)
                self.assertContinuesWithWarning(controller,observation)

    def test_missing_or_incomplete_native_metadata_warns_during_execution(self):
        for missing in ('snapshot','record','coroutines'):
            with self.subTest(missing=missing):
                controller,observation,_=self.setup_execution()
                if missing=='snapshot':
                    observation.pop('room_entities')
                elif missing=='record':
                    observation['room_entities']['entities']=[]
                else:
                    observation['room_entities']['entities'][0].pop('coroutines')
                self.assertContinuesWithWarning(controller,observation)

    def test_missing_or_duplicate_predicted_mechanism_warns_during_execution(self):
        for invalid in ('missing_key','missing_record','duplicate'):
            with self.subTest(invalid=invalid):
                controller,observation,_=self.setup_execution()
                predicted=controller.plan['expected'][0]
                if invalid=='missing_key':
                    predicted.pop('mechanisms')
                elif invalid=='missing_record':
                    predicted['mechanisms']=[]
                else:
                    predicted['mechanisms']*=2
                self.assertContinuesWithWarning(controller,observation)

    def test_player_dash_fields_are_not_required_to_validate_falling_environment(self):
        controller,observation,_=self.setup_execution()
        observation['state']=2
        observation['model_state']={'state':2}
        observation['model_state_unavailable']=['dash_cooldown','dash_buffer']
        self.assertEqual(controller.observe(observation),RIGHT)
        self.assertEqual(controller.mode,'executing')

    def test_ordinary_room_keeps_existing_execution_without_falling_metadata(self):
        controller,observation,_=self.setup_execution(falling=False)
        observation.pop('room_entities')
        controller.plan['expected'][0].pop('mechanisms',None)
        self.assertEqual(controller.observe(observation),RIGHT)
        self.assertEqual(controller.mode,'executing')


if __name__=='__main__':
    unittest.main()
