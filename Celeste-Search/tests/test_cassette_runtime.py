"""Fresh cassette telemetry and exact future-state admission, without a game."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist_validation import (
    _different_fields, _manual_state, _same, admit_assist_candidate, make_assist_anchor,
    cassette_comparison_snapshot, cassette_pickup_required,
)
from celeste_search.cassette import CassetteBlockState, CassetteClock
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DT, SimState, Simulator
from celeste_search.runtime_state import (RuntimeStateError, cassette_snapshot_from_observation,
                                         state_from_observation)
from celeste_search.routes import snapshot


def fixture():
    rows = ("0" * 16,) * 8 + ("1" * 16,) * 2
    room = Room("a", 0, 0, 128, 80, rows, (
        Entity("cassetteBlock", 1, 48, 48, 16, 8, {"index": 0}),
        Entity("cassetteBlock", 2, 64, 48, 16, 8, {"index": 0}),
    ), ((16.0, 64.0),))
    return MapData("cassette-runtime-fixture", {"a": room, "b": replace(room, name="b", entities=())},
                   Path("cassette-runtime-fixture.bin"))


def observed():
    state = SimState("a", 16.0, 64.0, on_ground=True, frame=100)
    value = state.to_dict()
    # The bridge exports environment separately, never inside model_state.
    model = {key: item for key, item in value.items()
             if key not in {"cassette_clock", "cassette_blocks", "cassette_stopped", "player_update_order",
                            "player_depth", "player_actual_depth"}}
    blocks = []
    for identity, x, order, leader, tracker in (("a:1", 48.0, 9, True, 0),
                                               ("a:2", 64.0, 10, False, 1)):
        blocks.append({"id": identity, "room": "a", "kind": "cassetteBlock", "present": True,
            "retained_observation": False, "unavailable": [], "index": 0, "tempo": 1.0,
            "x": x, "y": 48.0, "width": 16.0, "height": 8.0,
            "activated": True, "collidable": True, "active": True,
            "block_height": 2, "group_leader": leader, "group_ids": ["a:1", "a:2"],
            "update_order": order, "rem_y": 0.0, "lift_vx": 0.0, "lift_vy": -60.0,
            "depth": -10, "actual_depth": -10.0 - tracker * 0.000001,
            "tracker_order": tracker})
    value.update(model_state=model, model_state_unavailable=[],
        input_phase="before_virtual_inputs", control_mode="manual", human_input_mask=0,
        human_menu_input=False, input_bindings_supported=True, delta_time=DT, raw_delta_time=DT,
        human_input_epoch=1, spawn_id=1, deaths=0, on_safe_ground=True,
        room_entities={"schema_version": "celeste-room-entities-v1", "entities": blocks,
            "do_not_load": [], "unavailable": [], "player_update_order": 7,
            "player_depth": 0, "player_actual_depth": -0.000004,
            "depth_counters": {"0": 0.000005, "-10": 0.000002},
            "cassette_manager": {"present": True, "unavailable": [], "current_index": 0,
                "beat_timer": 0.082341, "beat_index": 8, "tempo_mult": 1.0,
                "lead_beats": 0, "max_beat": 2, "is_level_music": False,
                "beat_index_offset": 0, "active": True, "music_ready": True,
                "update_order": 8, "depth": 0, "actual_depth": -0.000003}})
    return value


def stopped_observed(height=1):
    value=observed()
    value['area_mode']=0
    room=value['room_entities']
    room['cassette_manager']={'present':False,'unavailable':[]}
    room['depth_counters']['8990']=.000003
    for index,block in enumerate(room['entities']):
        block.update(activated=False,collidable=False,block_height=height,y=50.-height,
                     depth=8990,actual_depth=8990.-index*.000001,lift_vx=0.,lift_vy=0.)
    return value


def observed_state(state):
    value = observed()
    encoded = state.to_dict()
    value.update(encoded)
    value["model_state"] = {key: item for key, item in encoded.items()
        if key not in {"cassette_clock", "cassette_blocks", "player_update_order", "player_depth", "player_actual_depth"}}
    snapshot = value["room_entities"]
    snapshot.update(player_update_order=state.player_update_order, player_depth=state.player_depth,
                    player_actual_depth=state.player_actual_depth,
                    depth_counters=state.cassette_clock.to_dict()["depth_counters"],
                    cassette_manager={**state.cassette_clock.to_dict(), "unavailable": []})
    snapshot["entities"] = [{**original, **block.to_dict(), "lift_vx": block.lift_x, "lift_vy": block.lift_y}
                            for original, block in zip(snapshot["entities"], state.cassette_blocks)]
    return value


class CassetteRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.map = fixture()
        self.obs = observed()

    def state(self, observation=None, previous=None):
        return state_from_observation(self.map, self.obs if observation is None else observation, previous)

    def test_explicit_absent_manager_and_complete_inactive_blocks_certify_stopped(self):
        for height in (0,1,2):
            with self.subTest(height=height):
                obs=stopped_observed(height)
                state=_manual_state(self.map,obs)
                self.assertTrue(state.cassette_stopped)
                self.assertIsNone(state.cassette_clock)
                self.assertEqual(len(state.cassette_blocks),2)
                self.assertTrue(all(block.block_height==height for block in state.cassette_blocks))
                simulator=Simulator(self.map)
                self.assertTrue(simulator.cassette_ready(state,state.room))
                anchor=make_assist_anchor(simulator,obs,request_id=4)
                self.assertEqual(anchor['mode'],'idle')
                self.assertTrue(admit_assist_candidate(self.map,anchor,obs,request_id=4)['accepted'])

    def test_missing_unknown_or_retained_manager_cannot_mean_stopped(self):
        for manager in (None,{}, {'unavailable':[]}, {'present':False},
                {'present':False,'unavailable':['present']},
                {'present':False,'unavailable':[],'retained_observation':True},
                {'present':False,'unavailable':[],'active':True}):
            obs=stopped_observed()
            obs['room_entities']['cassette_manager']=manager
            with self.subTest(manager=manager),self.assertRaises(RuntimeStateError):
                self.state(obs)
        obs=stopped_observed()
        del obs['room_entities']['cassette_manager']
        with self.assertRaises(RuntimeStateError):
            self.state(obs)

    def test_stopped_requires_every_fresh_inactive_stationary_block(self):
        for changes in ({'activated':True},{'collidable':True},{'active':False},
                {'present':False},{'retained_observation':True},{'unavailable':['activated']},
                {'depth':-10},{'y':50.},{'x':49.},{'block_height':3},{'rem_y':.01},
                {'lift_vx':1.},{'lift_vy':1.}):
            obs=stopped_observed()
            obs['room_entities']['entities'][0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(RuntimeStateError):
                self.state(obs)
        for name in CassetteBlockState.__dataclass_fields__:
            obs=stopped_observed()
            obs['room_entities']['entities'][0].pop({'lift_x':'lift_vx','lift_y':'lift_vy'}.get(name,name))
            with self.subTest(missing=name),self.assertRaises(RuntimeStateError):
                self.state(obs)
        obs=stopped_observed()
        obs['room_entities']['entities'].pop()
        with self.assertRaises(RuntimeStateError):
            self.state(obs)

    def test_stopped_is_room_local_and_never_inherited_over_fresh_evidence(self):
        stopped=self.state(stopped_observed())
        self.assertFalse(self.state(self.obs,stopped).cassette_stopped)
        outside=deepcopy(self.obs)
        outside['room']='b'
        reset=self.state(outside,stopped)
        self.assertFalse(reset.cassette_stopped)
        self.assertEqual(reset.cassette_blocks,())

    def test_stopped_comparison_preserves_geometry_and_strict_block_order(self):
        state=self.state(stopped_observed())
        raw=cassette_snapshot_from_observation(self.map,stopped_observed())
        self.assertTrue(raw['cassette_stopped'])
        canonical=cassette_comparison_snapshot(raw)
        self.assertNotIn('@manager',canonical['cassette_order'])
        self.assertEqual(canonical,cassette_comparison_snapshot(snapshot(state)))
        changed=replace(state,cassette_blocks=(replace(state.cassette_blocks[0],y=48.),state.cassette_blocks[1]))
        self.assertIn('cassette_blocks',_different_fields(changed,state,idle=False))
        blocks=state.cassette_blocks
        switched=replace(state,cassette_blocks=(replace(blocks[0],actual_depth=blocks[1].actual_depth),
                                                replace(blocks[1],actual_depth=blocks[0].actual_depth)))
        self.assertIn('cassette_order',_different_fields(switched,state,idle=False))
        dash=stopped_observed()
        dash['state']=2
        self.assertEqual(cassette_snapshot_from_observation(self.map,dash),raw)

    def test_stopped_scope_requires_normal_a_side_and_consistent_optional_session_flag(self):
        for changes in ({'area_mode':1},{'area_mode':None},{'state':11},{'state':True},
                {'in_cutscene':True},{'transitioning':True},{'session_cassette':False}):
            obs=stopped_observed()
            obs.update(changes)
            with self.subTest(changes=changes),self.assertRaises(RuntimeStateError):
                cassette_snapshot_from_observation(self.map,obs)

    def test_stopped_state_only_releases_pickup_guard_in_its_own_room(self):
        pickup=Entity('cassette',12,96,32,0,0)
        room=replace(self.map.rooms['a'],entities=self.map.rooms['a'].entities+(pickup,))
        data=replace(self.map,rooms={'a':room,'b':replace(room,name='b')})
        target={'room':'a','x':96,'y':32,'tile_size':8}
        stopped=state_from_observation(data,stopped_observed())
        self.assertTrue(cassette_pickup_required(data,target))
        self.assertFalse(cassette_pickup_required(data,target,stopped))
        self.assertTrue(cassette_pickup_required(data,{**target,'room':'b'},stopped))
        self.assertTrue(cassette_pickup_required(data,target,self.state()))

    def test_complete_live_clock_blocks_and_update_depth_are_preserved(self):
        state = self.state()
        self.assertEqual(state.cassette_clock.beat_timer, 0.082341)
        self.assertEqual(state.cassette_clock.beat_index, 8)
        self.assertEqual(state.cassette_clock.depth_counters, ((-10, 0.000002), (0, 0.000005)))
        self.assertEqual(state.player_update_order, 7)
        self.assertEqual(state.player_depth, 0)
        self.assertEqual(state.player_actual_depth, -0.000004)
        self.assertEqual([block.id for block in state.cassette_blocks], ["a:1", "a:2"])
        self.assertEqual(state.cassette_blocks[0].lift_y, -60.0)
        self.assertEqual(state.cassette_blocks[0].group_ids, ("a:1", "a:2"))
        self.assertEqual(state.x, 16.0)

    def test_environment_is_not_required_inside_private_player_model(self):
        state = _manual_state(self.map, self.obs)
        self.assertIsInstance(state.cassette_clock, CassetteClock)
        self.assertEqual(len(state.cassette_blocks), 2)

    def test_missing_each_clock_or_sorting_field_is_rejected(self):
        for name in CassetteClock.__dataclass_fields__:
            value = deepcopy(self.obs)
            parent = value["room_entities"] if name == "depth_counters" else value["room_entities"]["cassette_manager"]
            parent.pop(name)
            with self.subTest(field=name), self.assertRaises(RuntimeStateError):
                self.state(value)
        for name in ("player_update_order", "player_depth", "player_actual_depth"):
            value = deepcopy(self.obs)
            value["room_entities"].pop(name)
            with self.subTest(field=name), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_missing_block_fields_and_unknown_or_retained_records_are_rejected(self):
        for name in CassetteBlockState.__dataclass_fields__:
            value = deepcopy(self.obs)
            raw_name = {"lift_x": "lift_vx", "lift_y": "lift_vy"}.get(name, name)
            value["room_entities"]["entities"][0].pop(raw_name)
            with self.subTest(field=name), self.assertRaises(RuntimeStateError):
                self.state(value)
        for changes in ({"present": False}, {"active": False}, {"retained_observation": True},
                        {"retained_observation": None}, {"unavailable": ["activated"]},
                        {"unavailable": None}, {"kind": "unknown"}, {"room": "b"},
                        {"id": "a:unknown"}, {"width": 8}, {"index": 1}):
            value = deepcopy(self.obs)
            value["room_entities"]["entities"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_incomplete_groups_duplicate_ids_and_orders_are_rejected(self):
        for change in ("missing", "duplicate", "group", "order", "player_order", "tracker"):
            value = deepcopy(self.obs)
            snapshot = value["room_entities"]
            blocks = snapshot["entities"]
            if change == "missing":
                blocks.pop()
            elif change == "duplicate":
                blocks.append(deepcopy(blocks[0]))
            elif change == "group":
                blocks[0]["group_ids"] = ["a:1", "a:unknown"]
            elif change == "order":
                blocks[1]["update_order"] = blocks[0]["update_order"]
            elif change == "player_order":
                snapshot["player_update_order"] = blocks[0]["update_order"]
            else:
                blocks[1]["tracker_order"] = blocks[0]["tracker_order"]
            with self.subTest(change=change), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_absent_inactive_and_unknown_manager_are_never_guessed(self):
        previous = self.state()
        for changes in ({"present": False}, {"active": False}, {"retained_observation": True},
                        {"unavailable": ["beat_timer"]}, {"beat_timer": float("nan")}):
            value = deepcopy(self.obs)
            value["room_entities"]["cassette_manager"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeStateError):
                self.state(value, previous)
        for snapshot in (None, {"entities": []}):
            value = deepcopy(self.obs)
            value["room_entities"] = snapshot
            value["model_state"].update(previous.to_dict())
            with self.subTest(snapshot=snapshot), self.assertRaises(RuntimeStateError):
                self.state(value, previous)

    def test_non_cassette_room_clears_previous_phase_without_new_requirements(self):
        previous = self.state()
        value = deepcopy(self.obs)
        value.update(room="b", room_entities={"entities": []})
        state = self.state(value, previous)
        self.assertIsNone(state.cassette_clock)
        self.assertEqual(state.cassette_blocks, ())
        self.assertEqual(state.player_update_order, -1)
        self.assertEqual(state.player_actual_depth, 0.0)

    def test_invalid_player_depth_or_unavailable_order_is_rejected(self):
        for changes in ({"player_depth": True}, {"player_depth": 0.5},
                        {"player_actual_depth": float("nan")}, {"player_actual_depth": True},
                        {"unavailable": ["depth_counters"]}):
            value = deepcopy(self.obs)
            value["room_entities"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_partial_or_invalid_scene_depth_counters_are_rejected(self):
        for counters in ({"0": 0.000005}, {"0": -1, "-10": 0.000002},
                         {"0.5": 0.000005, "-10": 0.000002}, {"0": True}, []):
            value = deepcopy(self.obs)
            value["room_entities"]["depth_counters"] = counters
            with self.subTest(counters=counters), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_phase_and_depth_comparison_rejects_discrete_changes(self):
        state = self.state()
        clock = state.cassette_clock
        self.assertTrue(_same(clock, replace(clock, beat_timer=clock.beat_timer+0.00001)))
        self.assertFalse(_same(clock, replace(clock, beat_index=clock.beat_index+1)))
        self.assertFalse(_same(clock, replace(clock, active=False)))
        self.assertFalse(_same(clock, replace(clock, actual_depth=clock.actual_depth+0.000001)))
        self.assertFalse(_same(clock, replace(clock, depth_counters=((-10, 0.000003), (0, 0.000005)))))
        self.assertIn("cassette_order", _different_fields(
            state, replace(state, player_actual_depth=state.player_actual_depth+0.000002), idle=False))
        self.assertIn("cassette_clock", _different_fields(
            state, replace(state, cassette_clock=replace(clock, beat_index=9)), idle=False))
        self.assertFalse(_same(1, 1.00001))
        self.assertFalse(_same(True, 1))

    def test_decorative_allocation_gaps_preserve_admission_but_order_swaps_do_not(self):
        state = self.state()
        gap = replace(state,
            cassette_clock=replace(state.cassette_clock, depth_counters=((-10, 0.000003), (0, 0.000005))),
            cassette_blocks=tuple(replace(block, actual_depth=block.actual_depth-0.000001)
                                  for block in state.cassette_blocks))
        self.assertEqual(_different_fields(state, gap, idle=False), [])
        self.assertTrue(_same(cassette_comparison_snapshot(snapshot(state)),
                              cassette_comparison_snapshot(snapshot(gap))))
        first, second = gap.cassette_blocks
        swapped = replace(gap, cassette_blocks=(replace(first, actual_depth=second.actual_depth),
                                                replace(second, actual_depth=first.actual_depth)))
        self.assertIn("cassette_order", _different_fields(state, swapped, idle=False))

    def test_invalid_depth_buckets_cannot_use_order_equivalence(self):
        for changes in ({"player_actual_depth": -1.01},
                        {"player_actual_depth": 0.000001},
                        {"player_actual_depth": -0.1},
                        {"depth_counters": {"-10": 1.01, "0": 0.000005}}):
            value = deepcopy(self.obs)
            value["room_entities"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeStateError):
                self.state(value)

    def test_unrelated_scene_index_changes_do_not_override_exact_depth_order(self):
        state = self.state()
        changed = replace(state,
            player_update_order=state.player_update_order+20,
            cassette_clock=replace(state.cassette_clock, update_order=28),
            cassette_blocks=tuple(replace(block, update_order=block.update_order+20)
                                  for block in state.cassette_blocks))
        self.assertEqual(_different_fields(state, changed, idle=False), [])
        changed = replace(changed, cassette_blocks=(replace(changed.cassette_blocks[0], tracker_order=1),
                                                   replace(changed.cassette_blocks[1], tracker_order=0)))
        self.assertIn("cassette_blocks", _different_fields(state, changed, idle=False))

    def test_future_admission_requires_exact_frame_and_full_cassette_phase(self):
        simulator = Simulator(self.map)
        anchor = make_assist_anchor(simulator, self.obs, request_id="cassette", future_frames=2)
        waiting = admit_assist_candidate(self.map, anchor, self.obs, request_id="cassette")
        self.assertTrue(waiting["waiting"])
        expected = SimState.from_dict(anchor["state"])
        admitted = admit_assist_candidate(self.map, deepcopy(anchor), observed_state(expected),
            request_id="cassette", candidate_state=anchor["state"])
        self.assertTrue(admitted["accepted"], admitted)
        changed = replace(expected, cassette_clock=replace(expected.cassette_clock,
                          beat_index=(expected.cassette_clock.beat_index+1) % 256))
        rejected = admit_assist_candidate(self.map, deepcopy(anchor), observed_state(changed),
            request_id="cassette", candidate_state=anchor["state"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("cassette_clock", rejected["different_fields"])

    def test_execution_snapshot_is_strict_during_dash_without_anchor_defaults(self):
        expected = snapshot(self.state())
        observation = deepcopy(self.obs)
        observation.update(state=2, control_mode="assist", model_state={}, model_state_unavailable=["dash_frames"])
        actual = cassette_snapshot_from_observation(self.map, observation)
        self.assertEqual(actual, {key: expected[key] for key in actual})
        self.assertNotIn("update_order", actual["cassette_clock"])
        self.assertTrue(all("update_order" not in block for block in actual["cassette_blocks"]))
        self.assertEqual(cassette_snapshot_from_observation(self.map, {"room": "b", "state": 2}), {})

    def test_execution_snapshot_rejects_missing_phase_and_missing_block_during_dash(self):
        for change in ("manager", "block"):
            observation = deepcopy(self.obs)
            observation.update(state=2, model_state={})
            if change == "manager":
                observation["room_entities"]["cassette_manager"] = {"present": False, "unavailable": []}
            else:
                observation["room_entities"]["entities"].pop()
            with self.subTest(change=change), self.assertRaises(RuntimeStateError):
                cassette_snapshot_from_observation(self.map, observation)


if __name__ == "__main__":
    unittest.main()
