"""Observed moving solids replace static map ghosts without predicting motion."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from celeste_search.assist_gui import (AssistWindow,FALLING_REFRESH_MS,
                                     falling_map_geometry,falling_spike_owners)
from celeste_search.maps import Entity,MapData,Room


def fixture():
    block=Entity("fallingBlock",1,40.,16.,16.,16.)
    spikes=(Entity("spikesUp",2,40,16,16,0),Entity("spikesDown",3,40,32,16,0),
            Entity("spikesLeft",4,40,16,0,16),Entity("spikesRight",5,56,16,0,16))
    unrelated=Entity("spikesUp",6,80,80,16,0)
    room=Room("a",0,0,128,96,("0"*16,)*10+("1"*16,)*2,
              (block,*spikes,unrelated),((112.,80.),))
    return block,room,MapData("falling-gui",{"a":room},Path("fixture.bin"))


def record(native=8,**changes):
    value=dict(id="a:1",room="a",kind="fallingBlock",x=40.,y=32.,width=16.,height=16.,
        present=True,retained_observation=False,active=True,collidable=True,
        has_started_falling=True,triggered=False,fall_delay=0.,rem_x=0.,rem_y=0.,
        lift_vx=0.,lift_vy=0.,unavailable=["coroutine_phase"],coroutines=[dict(
            active=True,finished=False,wait_timer=0.,stack=[dict(
                iterator_type="Celeste.FallingBlock+<Sequence>d__21",fields={
                    "<>1__state":native,"<timer>5__4":.4,"<speed>5__2":40.,
                    "<maxSpeed>5__3":160.})])])
    value.update(changes)
    return value


def observation(raw=None):
    return dict(room="a",x=112,y=80,room_entities=dict(entities=[record() if raw is None else raw]))


class Canvas:
    def __init__(self):self.items=[];self.deleted=[];self.lowered=[]
    @staticmethod
    def tagged(item,tag):
        tags=item[2].get("tags",())
        return tag==tags if isinstance(tags,str) else tag in tags
    def delete(self,tag):
        self.deleted.append(tag)
        self.items=[] if tag=="all" else [item for item in self.items if not self.tagged(item,tag)]
    def find_withtag(self,tag):return [item for item in self.items if self.tagged(item,tag)]
    def create_rectangle(self,*args,**kwargs):self.items.append(("rectangle",args,kwargs))
    def create_text(self,*args,**kwargs):self.items.append(("text",args,kwargs))
    def create_line(self,*args,**kwargs):self.items.append(("line",args,kwargs))
    def tag_lower(self,*args):self.lowered.append(args)
    def winfo_width(self):return 500
    def winfo_height(self):return 300
    def xview_moveto(self,value):pass
    def yview_moveto(self,value):pass
    def configure(self,**kwargs):pass


class FallingMapTests(unittest.TestCase):
    def setUp(self):self.block,self.room,self.data=fixture()

    def geometry(self,obs):return falling_map_geometry(self.data,"a",obs)[0]

    def test_actual_collision_position_and_all_attached_spikes_move_together(self):
        raw=record(x=48.,y=40.,rem_x=.125,rem_y=-.25)
        obs=observation(raw);before=deepcopy(obs)
        item=self.geometry(obs)
        self.assertTrue(item.known)
        self.assertEqual(item.phase,2)
        self.assertEqual(item.rect,(48,40,16,16))
        self.assertEqual(item.hazards,((48,37,16,3),(48,56,16,3),(45,40,3,16),(64,40,3,16)))
        self.assertEqual(obs,before)

    def test_recorded_native_first_fall_does_not_draw_pending_subpixel_motion(self):
        evidence=json.loads((Path(__file__).parent/"fixtures"/"falling-native-single.json").read_text())
        raw=evidence["record"]
        block=Entity("fallingBlock",6,raw["map_x"],raw["map_y"],raw["width"],raw["height"])
        room=Room("12",3408,-2912,416,232,(),(block,),())
        data=MapData("native",{"12":room},Path("native.bin"))
        item=falling_map_geometry(data,"12",dict(room="12",room_entities=dict(entities=[raw])))[0]
        self.assertTrue(item.known)
        self.assertNotEqual(evidence["predicted"][4],raw["y"])
        self.assertEqual(item.rect,(3520,-2888,24,40))

    def test_permanently_landed_block_stays_at_its_new_position(self):
        item=self.geometry(observation(record(5,y=64.,coroutines=[])))
        self.assertTrue(item.known)
        self.assertEqual(item.phase,3)
        self.assertEqual(item.rect,(40,64,16,16))
        self.assertEqual(item.hazards[0],(40,61,16,3))

    def test_explicit_absence_or_native_removed_phase_hides_body_and_attached_spikes(self):
        raws=[dict(id="a:1",room="a",present=False,retained_observation=True,x=40.,y=120.),
              record(6,y=120.,collidable=False)]
        for raw in raws:
            with self.subTest(raw=raw):
                item=self.geometry(observation(raw))
                self.assertTrue(item.known)
                self.assertEqual(item.phase,4)
                self.assertIsNone(item.rect)
                self.assertEqual(item.hazards,())

    def test_missing_stale_wrong_room_duplicate_or_unknown_snapshot_cannot_reuse_last_position(self):
        self.assertEqual(self.geometry(observation()).rect,(40,32,16,16))
        bad=[None,{},dict(room="b",room_entities=observation()["room_entities"]),
             observation(record(retained_observation=True)),observation(record(room="b")),
             observation(record(42)),observation(record(triggered=True)),
             dict(room="a",room_entities=dict(entities=[record(),record()]))]
        for obs in bad:
            with self.subTest(obs=obs):
                item=self.geometry(obs)
                self.assertFalse(item.known)
                self.assertIsNone(item.phase)
                self.assertEqual(item.rect,(40,16,16,16))
                self.assertEqual(item.hazards[0],(40,13,16,3))

    def test_incomplete_multiple_block_snapshot_is_unknown_not_partial_live_state(self):
        room=replace(self.room,entities=(*self.room.entities,replace(self.block,id=9,x=96.)))
        data=replace(self.data,rooms={"a":room})
        items=falling_map_geometry(data,"a",observation())
        self.assertEqual(len(items),2)
        self.assertTrue(all(not item.known for item in items))

    def test_attachment_ownership_uses_first_mover_like_the_physics_model(self):
        zip_mover=replace(self.block,kind="zipMover",id=8)
        room=replace(self.room,entities=(zip_mover,*self.room.entities))
        self.assertEqual(falling_spike_owners(room),{})


class FallingCanvasTests(unittest.TestCase):
    def setUp(self):
        _,room,data=fixture()
        self.window=window=object.__new__(AssistWindow)
        window.map=data;window.room=room;window.canvas=Canvas()
        window.scale=2;window.offset=(20,20)
        window.shared=SimpleNamespace(snapshot=Mock(return_value=dict(connected=True,observation=observation())))
        window.root=Mock();window.root.after.return_value="timer"
        window._closed=False;window._falling_after=None

    def test_static_map_does_not_leave_original_falling_body_or_attached_spikes(self):
        window=self.window
        window.draw_search_previews=Mock();window.draw_dynamic=Mock()
        window.draw_map(fit=True)
        hazards=window.canvas.find_withtag("hazard")
        self.assertEqual(len(hazards),1)  # Only the unrelated static spike.
        self.assertFalse(any(item[2].get("fill")=="#647c91" for item in window.canvas.items))
        window.draw_falling()
        self.assertEqual(len(window.canvas.find_withtag("falling_body")),1)
        self.assertEqual(len(window.canvas.find_withtag("hazard")),5)

    def test_refresh_replaces_old_canvas_position_and_removal_clears_attached_hazards(self):
        window=self.window
        window.canvas.create_rectangle(0,0,1,1,tags="base")
        window.draw_falling()
        first=window.canvas.find_withtag("falling_body")[0][1]
        state=dict(connected=True,observation=observation(record(y=48.)))
        window.draw_falling(state)
        self.assertEqual(len(window.canvas.find_withtag("falling_body")),1)
        second=window.canvas.find_withtag("falling_body")[0][1]
        self.assertEqual(second[1]-first[1],32)
        removed=dict(id="a:1",room="a",present=False,x=40.,y=120.)
        window.draw_falling(dict(connected=True,observation=observation(removed)))
        self.assertEqual(window.canvas.find_withtag("falling"),[])
        self.assertEqual(len(window.canvas.find_withtag("base")),1)

    def test_disconnect_or_viewing_another_room_clears_live_geometry(self):
        window=self.window
        for state in (dict(connected=False,observation=observation()),
                      dict(connected=True,observation=dict(room="b")),
                      dict(connected=True,observation=dict(room="a")),
                      dict(connected=True,map_data=replace(window.map),observation=observation())):
            window.draw_falling()
            window.draw_falling(state)
            body=next(item for item in window.canvas.find_withtag("falling_body") if item[0]=="rectangle")
            self.assertEqual(body[1],(*window.point(40,16),*window.point(56,32)))
            self.assertEqual(body[2]["fill"],"")
            self.assertEqual(body[2]["dash"],(3,3))
            self.assertEqual(len([item for item in window.canvas.items if item[0]=="text"]),1)

    def test_unchanged_geometry_avoids_canvas_churn(self):
        window=self.window
        window.draw_falling()
        count=len(window.canvas.deleted)
        window.draw_falling()
        self.assertEqual(len(window.canvas.deleted),count)

    def test_fast_loop_only_updates_dynamic_layer_and_slows_when_not_live(self):
        window=self.window
        window.draw_map=Mock();window.draw_dynamic=Mock();window.draw_search_previews=Mock()
        window.refresh_falling()
        window.root.after.assert_called_once_with(FALLING_REFRESH_MS,window.refresh_falling)
        window.draw_map.assert_not_called();window.draw_dynamic.assert_not_called()
        window.draw_search_previews.assert_not_called()
        window.shared.snapshot.return_value=dict(connected=False)
        window.refresh_falling()
        self.assertEqual(window.root.after.call_args.args[0],250)
        window._closed=True;window.root.after.reset_mock()
        window.refresh_falling()
        window.root.after.assert_not_called()

    def test_close_cancels_dynamic_timer(self):
        window=self.window
        window._falling_after="timer";window.search_hover=Mock()
        window.shared.commands=Mock();window.shared.stop=Mock()
        window.close()
        self.assertTrue(window._closed)
        window.root.after_cancel.assert_called_once_with("timer")
        window.root.destroy.assert_called_once()


if __name__=="__main__":unittest.main()
