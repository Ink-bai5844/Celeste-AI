"""Observed platforms replace map ghosts without simulation or game input."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from celeste_search.assist_gui import AssistWindow, FALLING_REFRESH_MS, MAP_OBSERVATION_MAX_AGE
from celeste_search.assist_map_geometry import moving_map_geometry, moving_spike_owners
from celeste_search.maps import Entity, MapData, Room
from test_assist_falling_gui import Canvas
from test_cassette_runtime import fixture as cassette_fixture, stopped_observed


def fixture(kind="zipMover", dx=0, dy=0):
    mover=Entity(kind,1,40+dx,16+dy,16,16)
    spikes=(Entity("spikesUp",2,40+dx,16+dy,16,0),
            Entity("spikesDown",3,40+dx,32+dy,16,0),
            Entity("spikesLeft",4,40+dx,16+dy,0,16),
            Entity("spikesRight",5,56+dx,16+dy,0,16))
    room=Room("a",dx,dy,128,96,("0"*16,)*12,(mover,*spikes),((112.+dx,80.+dy),))
    return mover,room,MapData("moving-gui",{"a":room},Path("fixture.bin"))


def record(kind="zipMover",**changes):
    raw=dict(id="a:1",room="a",kind=kind,x=48.,y=40.,left=48.,top=40.,width=16.,height=16.,
             present=True,active=True,collidable=True,retained_observation=False,
             rem_x=.49,rem_y=-.4,unavailable=["coroutine_phase"])
    raw.update(changes)
    return raw


def observation(raw=None,**changes):
    obs=dict(room="a",frame=100,engine_frame=200,x=112,y=80,
             room_entities=dict(entities=[record() if raw is None else raw]))
    obs.update(changes)
    return obs


def window(data,room,obs):
    value=object.__new__(AssistWindow)
    value.map=data;value.room=room;value.canvas=Canvas()
    value.scale=2;value.offset=(20,20)
    value.shared=SimpleNamespace(snapshot=Mock(return_value=dict(connected=True,observation=obs)))
    value.root=Mock();value._closed=False;value._falling_after=None
    return value


class MovingGeometryTests(unittest.TestCase):
    def test_zip_collision_geometry_and_four_spike_faces_follow_actual_position(self):
        for dx,dy in ((0,0),(-136,728)):
            with self.subTest(translation=(dx,dy)):
                _,_,data=fixture(dx=dx,dy=dy)
                obs=observation(record(x=48+dx,y=40+dy,left=48+dx,top=40+dy))
                before=deepcopy(obs)
                item=moving_map_geometry(data,"a",obs)[0]
                self.assertTrue(item.known and item.solid)
                self.assertEqual(item.rect,(48+dx,40+dy,16,16))
                self.assertEqual(item.hazards,((48+dx,37+dy,16,3),(48+dx,56+dy,16,3),
                                               (45+dx,40+dy,3,16),(64+dx,40+dy,3,16)))
                self.assertEqual(obs,before)

    def test_real_native_outgoing_target_and_returning_zip_match_attached_spike_telemetry(self):
        evidence=json.loads((Path(__file__).parent/"fixtures"/"zip-native-minimap.json").read_text())
        mover=Entity("zipMover",2,824,-424,16,16)
        spike=Entity("spikesUp",3,824,-424,16,0)
        room=Room("02b",736,-576,384,216,(),(mover,spike),())
        data=MapData("native",{"02b":room},Path("native.bin"))
        positions=[]
        for row in evidence["rows"]:
            obs=row["observation"];raw=obs["room_entities"]["entities"]
            body=next(e for e in raw if e["kind"]=="zipMover")
            hazard=next(e for e in raw if e["kind"]=="spikesUp")
            item=moving_map_geometry(data,"02b",obs)[0]
            self.assertTrue(item.known)
            self.assertEqual(item.rect,tuple(body[k] for k in ("left","top","width","height")))
            self.assertEqual(item.hazards,(tuple(hazard[k] for k in ("left","top","width","height")),))
            positions.append(item.rect[:2])
        self.assertEqual(positions,[(824,-424),(826,-438),(832,-472),(832,-470),(827,-443)])

    def test_missing_duplicate_retained_wrong_room_or_bad_numbers_are_unknown(self):
        _,_,data=fixture()
        bad=[None,{},observation(room="b"),observation(record(room="b")),
             observation(record(retained_observation=True)),observation(record(left=float("nan"))),
             observation(record(top=True)),observation(record(width=0)),
             observation(record(left=10**1000)),
             observation(record(x=float("inf"))),observation(record(kind="fallingBlock")),
             observation(room_entities=dict(entities=[])),
             observation(room_entities=dict(entities=[record(),record()])),
             observation(room_entities=dict(room="b",entities=[record()])),
             observation(room_entities=dict(frame=99,entities=[record()]))]
        for obs in bad:
            with self.subTest(obs=obs):
                item=moving_map_geometry(data,"a",obs)[0]
                self.assertFalse(item.known or item.solid)
                self.assertEqual(item.rect,(40,16,16,16))

    def test_partial_group_never_blends_one_live_zip_with_one_unobserved_zip(self):
        mover,room,data=fixture()
        room=replace(room,entities=(*room.entities,replace(mover,id=10,x=96)))
        data=replace(data,rooms={"a":room})
        items=moving_map_geometry(data,"a",observation())
        self.assertEqual(len(items),2)
        self.assertTrue(all(not item.known for item in items))

    def test_explicit_absence_clears_body_and_owned_hazards_even_if_coordinates_retained(self):
        for kind in ("zipMover","dashBlock","crumbleBlock","cassetteBlock"):
            with self.subTest(kind=kind):
                _,_,data=fixture(kind)
                raw=dict(id="a:1",room="a",kind=kind,present=False,retained_observation=True)
                item=moving_map_geometry(data,"a",observation(raw))[0]
                self.assertTrue(item.known)
                self.assertIsNone(item.rect)
                self.assertEqual(item.hazards,())

    def test_crumble_and_dash_disappearance_then_reappearance_follow_collision_state(self):
        for kind in ("crumbleBlock","dashBlock"):
            _,_,data=fixture(kind)
            gone=moving_map_geometry(data,"a",observation(record(kind,collidable=False)))[0]
            restored=moving_map_geometry(data,"a",observation(record(kind)))[0]
            self.assertIsNone(gone.rect)
            self.assertTrue(restored.solid)
            self.assertEqual(restored.rect,(48,40,16,16))

    def test_cassette_inactive_phase_is_observed_outline_not_a_filled_static_platform(self):
        _,_,data=fixture("cassetteBlock")
        off=moving_map_geometry(data,"a",observation(record("cassetteBlock",collidable=False)))[0]
        on=moving_map_geometry(data,"a",observation(record("cassetteBlock")))[0]
        self.assertTrue(off.known and on.known)
        self.assertFalse(off.solid)
        self.assertTrue(on.solid)
        self.assertEqual(off.rect,on.rect)

    def test_only_strict_complete_stopped_cassette_snapshot_hides_all_blocks(self):
        data=cassette_fixture();obs=stopped_observed()
        for block in obs["room_entities"]["entities"]:
            block.update(left=block["x"],top=block["y"])
        items=moving_map_geometry(data,"a",obs)
        self.assertTrue(all(item.known and item.rect is None for item in items))
        missing=deepcopy(obs);missing["room_entities"]["entities"].pop()
        items=moving_map_geometry(data,"a",missing)
        self.assertTrue(all(not item.known and item.rect is not None for item in items))
        invalid=deepcopy(obs);invalid["room_entities"]["cassette_manager"]={}
        items=moving_map_geometry(data,"a",invalid)
        self.assertTrue(all(item.rect is not None for item in items))

    def test_first_attachment_owner_does_not_duplicate_spikes_between_overlapping_movers(self):
        mover,room,data=fixture()
        room=replace(room,entities=(mover,replace(mover,kind="dashBlock",id=9),*room.entities[1:]))
        data=replace(data,rooms={"a":room})
        obs=observation(room_entities=dict(entities=[record(),record("dashBlock",id="a:9")]))
        items=moving_map_geometry(data,"a",obs)
        self.assertEqual(len(moving_spike_owners(room)),4)
        self.assertEqual([len(item.hazards) for item in items],[4,0])


class MovingCanvasTests(unittest.TestCase):
    def setUp(self):
        _,self.room,self.data=fixture()
        self.window=window(self.data,self.room,observation())

    def test_base_has_no_zip_or_attached_spike_ghost_and_update_replaces_whole_layer(self):
        value=self.window
        value.draw_dynamic=Mock();value.draw_search_previews=Mock()
        value.draw_map(fit=True)
        self.assertEqual(value.canvas.find_withtag("hazard"),[])
        self.assertFalse(any(i[2].get("fill")=="#647c91" for i in value.canvas.items))
        value.draw_falling()
        first=value.canvas.find_withtag("mechanism_body")[0][1]
        value.draw_falling(dict(connected=True,observation=observation(record(x=64,y=56,left=64,top=56),frame=101)))
        bodies=value.canvas.find_withtag("mechanism_body")
        self.assertEqual(len(bodies),1)
        self.assertNotEqual(first,bodies[0][1])
        self.assertEqual(len(value.canvas.find_withtag("hazard")),4)
        self.assertNotIn("all",value.canvas.deleted[1:])

    def test_disconnect_map_room_or_missing_snapshot_clears_last_live_position(self):
        value=self.window
        for state in (dict(connected=False,observation=observation()),
                      dict(connected=True,map_data=replace(self.data),observation=observation()),
                      dict(connected=True,observation=observation(room="b")),
                      dict(connected=True,observation=observation(room_entities={}))):
            value.draw_falling(dict(connected=True,observation=observation()))
            value.draw_falling(state)
            body=next(i for i in value.canvas.find_withtag("mechanism_body") if i[0]=="rectangle")
            self.assertEqual(body[1],(*value.point(40,16),*value.point(56,32)))
            self.assertEqual(body[2]["fill"],"")
            self.assertEqual(body[2]["dash"],(3,3))

    def test_stalled_native_frame_becomes_unknown_and_new_frame_recovers(self):
        value=self.window
        with patch("celeste_search.assist_gui.time.monotonic",return_value=100) as clock:
            value.draw_falling()
            clock.return_value=100+MAP_OBSERVATION_MAX_AGE+.01
            value.draw_falling()
            self.assertTrue(any(i[0]=="text" and i[2].get("text")=="?" for i in value.canvas.items))
            value.draw_falling(dict(connected=True,observation=observation(frame=101,engine_frame=201)))
            self.assertFalse(any(i[0]=="text" for i in value.canvas.items))
            self.assertEqual(len(value.canvas.find_withtag("mechanism_body")),1)

    def test_cassette_live_color_is_cleared_on_disconnect_without_leaving_old_layer(self):
        _,room,data=fixture("cassetteBlock")
        value=window(data,room,observation(record("cassetteBlock")))
        value.draw_falling()
        self.assertTrue(value.canvas.find_withtag("cassette")[0][2]["fill"])
        value.draw_falling(dict(connected=False,observation=observation(record("cassetteBlock"))))
        body=next(i for i in value.canvas.find_withtag("cassette") if i[0]=="rectangle")
        self.assertEqual(body[2]["fill"],"")
        self.assertEqual(len([i for i in value.canvas.find_withtag("cassette") if i[0]=="rectangle"]),1)

    def test_zip_only_room_uses_fast_local_refresh_without_map_redraw_or_search_io(self):
        value=self.window
        value.draw_map=Mock();value.draw_dynamic=Mock();value.draw_search_previews=Mock()
        value.refresh_falling()
        value.root.after.assert_called_once_with(FALLING_REFRESH_MS,value.refresh_falling)
        value.draw_map.assert_not_called();value.draw_dynamic.assert_not_called()
        value.draw_search_previews.assert_not_called()

    def test_unchanged_geometry_does_not_redraw_and_explicit_removal_clears_hazards(self):
        value=self.window;value.draw_falling()
        deleted=len(value.canvas.deleted)
        value.draw_falling(dict(connected=True,observation=observation(frame=101)))
        self.assertEqual(len(value.canvas.deleted),deleted)
        value.draw_falling(dict(connected=True,observation=observation(record(present=False,retained_observation=True),frame=102)))
        self.assertEqual(value.canvas.find_withtag("mechanism"),[])


if __name__=="__main__":
    unittest.main()
