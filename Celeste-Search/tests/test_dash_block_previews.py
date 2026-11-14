"""Conditional dash-break sketches remain distinct from observed game geometry."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from celeste_search.assist_gui import corridor_caption, DASH_CONDITION_COLOR
from celeste_search.assist_map_geometry import moving_map_geometry
from celeste_search.physics import SimState
from celeste_search.search_progress import (dash_break_conditions, progress_reporter,
                                            publish_corridor_progress, search_tracker)
import test_assist_moving_gui as moving
import test_assist_search_previews_gui as previews


def condition(**changes):
    result=dict(kind="dash_break",identity="a:1",room="a",rect=[40,16,56,32],
                can_dash=True,approach=dict(x=32,y=28),dash_direction=dict(x=1,y=0))
    result.update(changes)
    return result


def preview_window():
    fixture=previews.PreviewWindowTests()
    fixture.setUp()
    window=fixture.window
    item=previews.topology_entry(physical=True)
    item["topology"]["corridors"][1]["conditions"]=[condition()]
    window.shared.update(search_previews=[item])
    return window,item


class DashBlockGeometryTests(unittest.TestCase):
    def test_dash_permission_uses_map_boolean_and_never_marks_forbidden_as_breakable(self):
        for raw,expected in ((True,True),("true",True),(False,False),("false",False),(0,False)):
            with self.subTest(canDash=raw):
                mover,room,data=moving.fixture("dashBlock")
                room=replace(room,entities=(replace(mover,attrs=dict(canDash=raw)),*room.entities[1:]))
                data=replace(data,rooms={"a":room})
                item=moving_map_geometry(data,"a",moving.observation(moving.record("dashBlock")))[0]
                self.assertEqual(item.dash_breakable,expected)
                self.assertEqual(item.color==DASH_CONDITION_COLOR,expected)
                window=moving.window(data,room,moving.observation(moving.record("dashBlock")))
                window.draw_falling()
                self.assertEqual(any(row[2].get("text")=="冲刺" for row in window.canvas.items),expected)

    def test_fresh_removal_erases_body_hint_and_owned_hazards_then_respawn_restores_them(self):
        _,room,data=moving.fixture("dashBlock")
        obs=moving.observation(moving.record("dashBlock"));before=deepcopy(obs)
        window=moving.window(data,room,obs)
        window.draw_falling()
        self.assertTrue(any(row[2].get("text")=="冲刺" for row in window.canvas.items))
        for raw in (moving.record("dashBlock",present=False,retained_observation=True),
                    moving.record("dashBlock",collidable=False)):
            window.draw_falling(dict(connected=True,observation=moving.observation(raw,frame=101)))
            self.assertEqual(window.canvas.find_withtag("mechanism"),[])
        window.draw_falling(dict(connected=True,observation=moving.observation(moving.record("dashBlock"),frame=102)))
        self.assertEqual(len(window.canvas.find_withtag("hazard")),4)
        self.assertTrue(any(row[2].get("text")=="冲刺" for row in window.canvas.items))
        self.assertEqual(obs,before)

    def test_fresh_native_permission_false_also_removes_breakable_hint(self):
        _,room,data=moving.fixture("dashBlock")
        observed=moving.observation(moving.record("dashBlock",can_dash=False))
        item=moving_map_geometry(data,"a",observed)[0]
        self.assertTrue(item.known and item.solid)
        self.assertFalse(item.dash_breakable)
        self.assertNotEqual(item.color,DASH_CONDITION_COLOR)

    def test_unknown_snapshot_retains_question_mark_without_live_break_instruction(self):
        _,room,data=moving.fixture("dashBlock")
        window=moving.window(data,room,moving.observation(room_entities={}))
        window.draw_falling()
        texts=[row[2].get("text") for row in window.canvas.items]
        self.assertIn("?",texts)
        self.assertNotIn("冲刺",texts)


class DashBlockPreviewTests(unittest.TestCase):
    def test_condition_metadata_is_bounded_copied_and_cannot_claim_controls_or_completion(self):
        start=SimState("a",16,96,on_ground=True)
        route=dict(waypoints=[dict(room="a",x=80,y=88)])
        values=[condition(identity=f"a:{index}") for index in range(30)]
        values[0].update(actions=[32],physically_verified=True)
        corridor=dict(id="c",status="pending",conditions=values,
                      points=[dict(room="a",x=16,y=96),dict(room="a",x=80,y=96)])
        packets=[]
        with progress_reporter(packets.append,start,route):
            publish_corridor_progress([corridor],"c")
            search_tracker(start).report(lambda:[("best",start,[start])],1,1)
        for packet in packets:
            rows=packet["topology"]["corridors"][0]["conditions"]
            self.assertEqual(len(rows),16)
            self.assertNotIn("actions",str(rows))
            self.assertNotIn("physically_verified",str(rows))
            self.assertEqual(packet["waypoints_reached"],0)
        values[0]["rect"][0]=999
        self.assertEqual(packets[-1]["topology"]["corridors"][0]["conditions"][0]["rect"][0],40)

    def test_false_permission_and_invalid_conditions_never_receive_dash_hint(self):
        malformed=[condition(can_dash=False),condition(can_dash=1),condition(rect=[1,2,0,5]),
                   condition(rect=[1,2,float("nan"),5]),condition(rect=[1,2,10**1000,5]),
                   condition(dash_direction=dict(x=0,y=0)),condition(dash_direction=dict(x=True,y=0)),
                   condition(approach=dict(x=float("inf"),y=0))]
        for row in malformed:
            corridor=dict(id="c",status="pending",conditions=[row])
            self.assertEqual(dash_break_conditions(corridor),[])
            self.assertNotIn("冲刺撞开",corridor_caption(corridor,1,1))

    def test_conditioned_geometry_is_dotted_even_when_verified_and_actual_path_is_separate(self):
        window,item=preview_window()
        item["topology"]["corridors"][1]["status"]="verified"
        window.shared.update(search_previews=[item])
        before=window.draft.payload()
        window.draw_search_previews(force=True)
        conditional=[row for row in window.canvas.items if row[0]=="line" and row[2]["fill"]==DASH_CONDITION_COLOR]
        self.assertTrue(conditional)
        self.assertTrue(all(row[2]["dash"]==(1,5) for row in conditional))
        color=window.preview_styles.get("1")[1]
        physical=[row for row in window.canvas.items if row[0]=="line" and row[2]["fill"]==color]
        self.assertTrue(physical)
        self.assertTrue(all("dash" not in row[2] for row in physical))
        self.assertTrue(any(row[2].get("text")=="冲刺撞开此处" for row in window.canvas.items))
        self.assertEqual(window.draft.payload(),before)
        self.assertTrue(window.shared.commands.empty())

    def test_fresh_native_removal_clears_old_condition_marker_without_waiting_for_preview_throttle(self):
        window,item=preview_window()
        _,room,data=moving.fixture("dashBlock")
        window.map,window.room=data,room
        present=moving.observation(moving.record("dashBlock"))
        state=dict(mode="planning",connected=True,search_generation=3,search_previews=[item],observation=present)
        with patch("celeste_search.assist_gui.time.monotonic",return_value=100.) as clock:
            window.draw_search_previews(state,force=True)
            self.assertIn("需冲刺撞开",window.preview_topology_status)
            clock.return_value=100.01
            state["observation"]=moving.observation(moving.record("dashBlock",present=False),frame=101)
            window.draw_search_previews(state)
        texts=[row[2].get("text","") for row in window.canvas.items]
        self.assertNotIn("冲刺撞开此处",texts)
        self.assertIn("撞开条件已满足",window.preview_topology_status)
        self.assertIn("几何待验证",window.preview_topology_status)
        self.assertNotIn("需冲刺撞开",window.preview_topology_status)
        self.assertFalse(any(row[0]=="line" and row[2]["fill"]==DASH_CONDITION_COLOR for row in window.canvas.items))
        # A missing or stale snapshot is not proof of destruction.
        state["observation"]=moving.observation(room_entities={},frame=102)
        window.draw_search_previews(state,force=True)
        self.assertIn("需冲刺撞开",window.preview_topology_status)

    def test_predicted_condition_never_hides_actual_block_and_other_room_has_no_marker(self):
        window,item=preview_window()
        _,room,data=moving.fixture("dashBlock")
        observed=moving.observation(moving.record("dashBlock"))
        physical=moving.window(data,room,observed)
        physical.draw_falling()
        before=deepcopy(physical.canvas.items)
        item["topology"]["corridors"][1]["conditions"][0]["room"]="other"
        window.shared.update(search_previews=[item])
        window.draw_search_previews(force=True)
        self.assertFalse(any(row[2].get("text")=="冲刺撞开此处" for row in window.canvas.items))
        self.assertEqual(physical.canvas.items,before)
        self.assertIsNotNone(moving_map_geometry(data,"a",observed)[0].rect)

    def test_old_condition_cannot_override_native_cannot_dash(self):
        window,item=preview_window()
        _,room,data=moving.fixture("dashBlock")
        window.map,window.room=data,room
        observed=moving.observation(moving.record("dashBlock",can_dash=False))
        state=dict(mode="planning",connected=True,search_generation=3,search_previews=[item],observation=observed)
        window.draw_search_previews(state,force=True)
        self.assertIn("冲刺破坏条件不成立",window.preview_topology_status)
        self.assertNotIn("需冲刺撞开",window.preview_topology_status)
        self.assertFalse(any(row[2].get("text")=="冲刺撞开此处" for row in window.canvas.items))
        self.assertFalse(any(row[0]=="line" and row[2]["fill"]==DASH_CONDITION_COLOR for row in window.canvas.items))


if __name__=="__main__":
    unittest.main()
