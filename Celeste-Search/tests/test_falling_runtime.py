"""Verified native FallingBlock phases and refusal of ambiguous telemetry."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import unittest

from celeste_search.assist_validation import AssistValidationError, make_assist_anchor
from celeste_search.falling_runtime import FallingRuntimeError, decode_falling_record
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DT, Simulator
from celeste_search.runtime_state import (UnsupportedFallingRuntimeState,
    falling_snapshot_from_observation, state_from_observation)


def fixture():
    block = Entity("fallingBlock", 1, 40., 16., 16., 16.)
    room = Room("a", 0, 0, 128, 96, ("0"*16,)*10+("1"*16,)*2,
                (block,), ((112.,80.),))
    return block, room, MapData("falling-runtime", {"a":room}, Path("falling-runtime.bin"))


def record(native=1, *, wait=0., timer=0., speed=0., **changes):
    value = dict(id="a:1", room="a", kind="fallingBlock", runtime_type="Celeste.FallingBlock",
        x=40., y=16., width=16., height=16., present=True, retained_observation=False,
        active=True, collidable=True, has_started_falling=native not in (0,1,2),
        triggered=False, fall_delay=0., rem_x=0., rem_y=0., lift_vx=0., lift_vy=0.,
        unavailable=["coroutine_phase"], coroutines=[dict(active=True, finished=False, wait_timer=wait,
            stack=[dict(iterator_type="Celeste.FallingBlock+<Sequence>d__21", fields={
                "<>1__state":native, "<timer>5__4":timer, "<speed>5__2":speed,
                "<maxSpeed>5__3":160. if native==8 else 0.})])])
    value.update(changes)
    return value


def observation(data, records):
    state = Simulator(data).initial_state("a")
    return dict(state.to_dict(), model_state=state.to_dict(), model_state_unavailable=[],
        input_phase="before_virtual_inputs", on_safe_ground=True, control_mode="manual",
        human_input_mask=0, human_menu_input=False, input_bindings_supported=True,
        delta_time=DT, raw_delta_time=DT, human_input_epoch=4, spawn_id=2, deaths=1,
        area_id=1, area_mode=0, room_entities=dict(entities=records, do_not_load=[], unavailable=[]))


class FallingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.block, self.room, self.data = fixture()

    def decode(self, raw):
        return decode_falling_record(raw, self.block, self.room)

    def test_idle_and_fall_delay_are_distinct(self):
        self.assertEqual(self.decode(record())[1:3], (0,0))
        self.assertEqual(self.decode(record(2, fall_delay=.0333334))[1:3], (7,2))
        self.assertEqual(self.decode(record(2, fall_delay=0))[1:3], (7,0))
        with self.assertRaises(FallingRuntimeError):
            self.decode(record(fall_delay=.1))  # Pending pre-trigger delay needs another persistent field.

    def test_behind_variant_is_refused_before_idle_anchor_admission(self):
        for value in (True, "true", 1):
            with self.subTest(value=value):
                block=replace(self.block,attrs={"behind":value})
                room=replace(self.room,entities=(block,))
                data=replace(self.data,rooms={"a":room})
                with self.assertRaises(AssistValidationError) as context:
                    make_assist_anchor(Simulator(data),observation(data,[record()]),request_id=1)
                self.assertEqual(context.exception.reason,"unsupported_falling_runtime_state")
                self.assertIn("behind=true",str(context.exception))

    def test_final_boss_marker_is_refused_before_idle_anchor_admission(self):
        block=replace(self.block,attrs={"finalBoss":True})
        room=replace(self.room,entities=(block,))
        data=replace(self.data,rooms={"a":room})
        with self.assertRaises(AssistValidationError) as context:
            make_assist_anchor(Simulator(data),observation(data,[record()]),request_id=1)
        self.assertEqual(context.exception.reason,"unsupported_falling_runtime_state")
        self.assertIn("finalBoss=true",str(context.exception))
        native=replace(self.block,kind="finalBossFallingBlock")
        with self.assertRaisesRegex(FallingRuntimeError,"finalBossFallingBlock"):
            decode_falling_record(record(),native,replace(self.room,entities=(native,)))

    def test_false_variant_flags_keep_ordinary_idle_import(self):
        for value in (False,"false",0):
            with self.subTest(value=value):
                block=replace(self.block,attrs={"behind":value,"finalBoss":value})
                self.assertEqual(decode_falling_record(record(),block,self.room)[1:3],(0,0))

    def test_initial_shake_preserves_the_native_resume_tick(self):
        for wait, elapsed in ((.200000003,0),(.0833331347,7),(-3.57627869e-7,12)):
            with self.subTest(wait=wait):
                self.assertEqual(self.decode(record(3,wait=wait))[1:3], (1,elapsed))

    def test_player_contact_wait_is_elapsed_and_never_idle(self):
        for remaining, elapsed in ((.400000006,13),(.333333164,17),(.183332846,26),(.0166660547,36)):
            with self.subTest(timer=remaining):
                self.assertEqual(self.decode(record(4,timer=remaining))[1:3], (1,elapsed))

    def test_fall_uses_iterator_speed_and_preserves_movement_remainders(self):
        decoded = self.decode(record(8,speed=41.666748,y=32.,rem_y=-.25,rem_x=.125,lift_vy=0.))
        self.assertEqual(decoded, ("a:1",2,0,40.125,31.75,41.666748046875))

    def test_live_first_fall_single_decimals_reproduce_exact_mechanism_tuple(self):
        fixture=json.loads((Path(__file__).parent/"fixtures"/"falling-native-single.json").read_text())
        raw=fixture["record"]
        block=Entity("fallingBlock",6,raw["map_x"],raw["map_y"],raw["width"],raw["height"])
        room=Room("12",3408,-2912,416,232,(),(block,),())
        self.assertEqual(list(decode_falling_record(raw,block,room)),fixture["predicted"])
        self.assertNotEqual(raw["y"]+raw["rem_y"],fixture["predicted"][4])

    def test_native_single_overflow_and_nonfinite_values_are_validation_errors(self):
        for key in ("x","y","rem_x","rem_y"):
            for value in (1e40,10**1000,float("inf"),float("nan")):
                with self.subTest(key=key,value_type=type(value).__name__),self.assertRaises(FallingRuntimeError):
                    self.decode(record(**{key:value}))
        for value in (1e40,10**1000,float("inf"),float("nan")):
            with self.subTest(speed=value if isinstance(value,float) else "huge int"),self.assertRaises(FallingRuntimeError):
                self.decode(record(8,speed=value))

    def test_landing_shake_and_platform_poll_keep_zero_resume_boundary(self):
        for native, phase, wait, ticks in ((5,5,.2,12),(5,5,0.,0),(9,6,.1,6),(9,6,-1e-7,0)):
            with self.subTest(native=native,wait=wait):
                self.assertEqual(self.decode(record(native,wait=wait,y=64.))[1:3],(phase,ticks))

    def test_repeat_shake_retains_the_native_iterator_speed_until_next_fall(self):
        for native, wait, timer in ((3,.2,0),(4,0,.4),(5,.2,0),(9,.1,0)):
            with self.subTest(native=native):
                self.assertEqual(self.decode(record(native,wait=wait,timer=timer,speed=160.))[5],160.)

    def test_only_tiles_support_permanent_completion(self):
        raw=record(5,y=64.,coroutines=[])
        self.assertEqual(self.decode(raw)[1:3],(3,0))
        raw["coroutines"]=[dict(active=False,finished=True,stack=[],wait_timer=0.)]
        self.assertEqual(self.decode(raw)[1:3],(3,0))
        empty=replace(self.room,solids=(),entities=(self.block,Entity("jumpThru",2,40,80,16,8)))
        with self.assertRaises(FallingRuntimeError):
            decode_falling_record(raw,self.block,empty)
        with self.assertRaises(FallingRuntimeError):
            self.decode(record(coroutines=[]))

    def test_verified_out_of_bounds_or_explicit_absence_is_removed(self):
        for native in (6,7):
            self.assertEqual(self.decode(record(native,wait=.1,collidable=False,y=120.))[1:3],(4,0))
        self.assertEqual(self.decode(dict(id="a:1",room="a",present=False,
            retained_observation=True,x=40.,y=120.)),("a:1",4,0,40.,120.,0.))
        with self.assertRaises(FallingRuntimeError):
            self.decode(record(collidable=False))

    def test_unknown_schema_and_unavailable_phase_fields_are_refused(self):
        examples=[]
        raw=record();raw.pop("coroutines");examples.append(raw)
        raw=record();raw["coroutines"][0]["stack"][0]["iterator_type"]="Celeste.FallingBlock+OtherBuild";examples.append(raw)
        raw=record(4);raw["coroutines"][0]["stack"][0]["fields"].pop("<timer>5__4");examples.append(raw)
        raw=record();raw["coroutines"][0]["stack"][0]["fields"]["<>1__state"]=True;examples.append(raw)
        examples.extend([record(42),record(triggered=True),record(retained_observation=True),
                         record(active=False),record(unavailable=["coroutine_stack"]),
                         record(4,timer=float("nan")),record(8,speed=180),record(has_started_falling=True)])
        for raw in examples:
            with self.subTest(raw=raw),self.assertRaises(FallingRuntimeError):
                self.decode(raw)

    def test_complete_native_snapshot_overrides_prior_idle_phase(self):
        prior=Simulator(self.data).initial_state("a")
        obs=observation(self.data,[record(4,timer=.183332846)])
        state=state_from_observation(self.data,obs,prior)
        self.assertEqual(state.mechanisms,(("a:1",1,26,40.,16.,0.),))
        self.assertEqual((state.x,state.y,state.frame),(prior.x,prior.y,prior.frame))

    def test_missing_duplicate_wrong_room_or_unknown_records_cannot_use_prior_prediction(self):
        prior=Simulator(self.data).initial_state("a")
        for rows in ([],[record(),record()],[record(room="b")],[record(42)]):
            with self.subTest(rows=rows),self.assertRaises(UnsupportedFallingRuntimeState):
                state_from_observation(self.data,observation(self.data,rows),prior)
        obs=observation(self.data,[record()]);obs.pop("room_entities")
        with self.assertRaises(UnsupportedFallingRuntimeState):
            state_from_observation(self.data,obs,prior)

    def test_unsupported_phase_is_an_assist_validation_error_not_a_crash(self):
        with self.assertRaises(AssistValidationError) as context:
            make_assist_anchor(Simulator(self.data),observation(self.data,[record(42)]),request_id=1)
        self.assertEqual(context.exception.reason,"unsupported_falling_runtime_state")

    def test_execution_snapshot_uses_native_phase_without_player_dash_fields(self):
        raw=record(4,timer=.183332846,rem_y=.125)
        obs=dict(room="a",state=2,model_state_unavailable=["dash_frames"],
                 room_entities=dict(entities=[raw]))
        self.assertEqual(falling_snapshot_from_observation(self.data,obs),
                         [["a:1",1,26,40.,16.125,0.]])
        raw["coroutines"][0]["stack"][0]["fields"]["<timer>5__4"] = .16666615
        self.assertEqual(falling_snapshot_from_observation(self.data,obs)[0][2],27)

    def test_execution_snapshot_requires_all_unique_fresh_falling_records(self):
        examples=[dict(room="a"),dict(room="a",room_entities={}),
                  dict(room="a",room_entities=dict(entities=[]))]
        for rows in ([record(),record()],[record(room="b")],
                     [record(retained_observation=True)],[record(coroutines=[])],
                     [record(x=float("nan"))]):
            examples.append(dict(room="a",room_entities=dict(entities=rows)))
        for obs in examples:
            with self.subTest(obs=obs),self.assertRaises(UnsupportedFallingRuntimeState):
                falling_snapshot_from_observation(self.data,obs)
        second=replace(self.block,id=2,x=64.)
        room=replace(self.room,entities=(second,self.block))
        data=replace(self.data,rooms={"a":room})
        raw2=record(id="a:2",x=64.)
        obs=dict(room="a",room_entities=dict(entities=[raw2,record()]))
        self.assertEqual([v[0] for v in falling_snapshot_from_observation(data,obs)],["a:1","a:2"])
        obs["room_entities"]["entities"].pop()
        with self.assertRaises(UnsupportedFallingRuntimeState):
            falling_snapshot_from_observation(data,obs)

    def test_execution_snapshot_does_not_require_telemetry_in_non_falling_room(self):
        room=replace(self.room,entities=())
        data=replace(self.data,rooms={"a":room})
        self.assertEqual(falling_snapshot_from_observation(data,dict(room="a",state=2)),[])


if __name__ == "__main__":
    unittest.main()
