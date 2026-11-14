"""Search candidates stay visual, bounded, and independent of confirmed routes."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest

from celeste_search.assist import SharedAssistState
from celeste_search.assist_gui import (AssistWindow,RouteDraft,PreviewStyles,
                                     preview_room_segments,current_search_previews,place_preview_label,
                                     topology_progress_label)


class Value:
    def __init__(self,value):
        self.value=value
    def get(self):
        return self.value
    def set(self,value):
        self.value=value


class Canvas:
    def __init__(self):
        self.items=[("draft",(),{"tags":"dynamic"})]
        self.line_calls=0
    def delete(self,tag):
        self.items=[item for item in self.items if item[2].get("tags")!=tag]
    def create_line(self,*args,**kwargs):
        self.line_calls+=1
        self.items.append(("line",args,kwargs))
    def create_oval(self,*args,**kwargs):
        self.items.append(("oval",args,kwargs))
    def create_text(self,*args,**kwargs):
        self.items.append(("text",args,kwargs))
    def find_withtag(self,tag):
        return [index for index,item in enumerate(self.items) if item[2].get("tags")==tag]
    def tag_lower(self,*args):
        pass
    def canvasx(self,value):return value
    def canvasy(self,value):return value
    def winfo_width(self):return 500
    def winfo_height(self):return 300


def entry(worker=1,*,updated=1,local=False):
    return dict(generation=3,worker_id=worker,strategy=f"strategy-{worker}",label=f"strategy {worker}",
                expansions=100,waypoints_reached=1,updated_at=updated,
                paths=[dict(kind="best",local_leg=local,points=[
                    dict(room="a",x=50,y=30),dict(room="a",x=60,y=40),
                    dict(room="b",x=70,y=40),dict(room="a",x=80,y=50),dict(room="a",x=90,y=60)])])


def topology_entry(*,physical=False):
    result=entry()
    result["paths"]=result["paths"] if physical else []
    result["waypoints_reached"]=0
    result["topology"]=dict(active_id="c2",corridors=[
        dict(id="c1",status="failed",reason="portal_unreachable",
             points=[dict(room="a",x=10,y=30),dict(room="a",x=40,y=50)]),
        dict(id="c2",status="trying",reason="",
             points=[dict(room="a",x=10,y=30),dict(room="a",x=30,y=70)]),
        dict(id="c3",status="pending",reason="",
             points=[dict(room="a",x=10,y=30),dict(room="a",x=70,y=30)])])
    return result


class PreviewHelpersTests(unittest.TestCase):
    def test_twelve_coincident_labels_are_separated_with_bounded_offsets(self):
        boxes=[]
        for _ in range(12):
            place_preview_label(250,150,28,16,boxes,(4,4,496,296))
        for index,box in enumerate(boxes):
            self.assertLessEqual(abs((box[0]+box[2])/2-250),80)
            self.assertLessEqual(abs((box[1]+box[3])/2-150),65)
            for other in boxes[:index]:
                self.assertFalse(box[0]<other[2] and box[2]>other[0] and box[1]<other[3] and box[3]>other[1])

    def test_labels_at_viewport_edge_remain_inside_visible_bounds(self):
        boxes=[]
        place_preview_label(495,5,60,18,boxes,(4,4,496,296))
        left,top,right,bottom=boxes[0]
        self.assertGreaterEqual(left,4)
        self.assertGreaterEqual(top,4)
        self.assertLessEqual(right,496)
        self.assertLessEqual(bottom,296)

    def test_distinct_workers_keep_stable_colors_when_strategies_change(self):
        styles=PreviewStyles()
        values=[styles.get(worker) for worker in range(12)]
        self.assertEqual(len({color for _,color in values}),12)
        self.assertEqual(styles.get(3),values[3])
        self.assertEqual(styles.get("3"),values[3])

    def test_room_returns_are_separate_segments_and_raw_feet_y_is_offset(self):
        pieces=preview_room_segments(entry()["paths"][0]["points"],"a")
        self.assertEqual(pieces,[[(50,25),(60,35)],[(80,45),(90,55)]])

    def test_long_paths_are_bounded_with_endpoints_preserved(self):
        points=[dict(room="a",x=index,y=20) for index in range(10000)]
        pieces=preview_room_segments(points,"a",120)
        self.assertEqual(sum(map(len,pieces)),120)
        self.assertEqual(pieces[0][0],(0,15))
        self.assertEqual(pieces[-1][-1],(9999,15))

    def test_invalid_points_break_a_path_instead_of_bridging_unknown_positions(self):
        points=[dict(room="a",x=1,y=5),dict(room="a",x=float("nan"),y=5),dict(room="a",x=10,y=5)]
        self.assertEqual(preview_room_segments(points,"a"),[[(1,0)],[(10,0)]])

    def test_stale_generations_and_nonsearch_modes_are_not_displayed(self):
        state=dict(mode="planning",search_generation=4,search_previews=[entry()])
        self.assertEqual(current_search_previews(state),[])
        state["search_generation"]=3
        self.assertEqual(len(current_search_previews(state)),1)
        for mode in ("executing","success","failed","manual"):
            state["mode"]=mode
            self.assertEqual(current_search_previews(state),[])

    def test_topology_only_entry_is_visible_and_still_filtered_by_generation(self):
        item=topology_entry();del item["paths"]
        state=dict(mode="planning",search_generation=3,search_previews=[item])
        self.assertEqual(len(current_search_previews(state)),1)
        self.assertIn("通路 2/3",topology_progress_label(current_search_previews(state)))
        self.assertIn("验证中",topology_progress_label([item]))
        self.assertIn("1 条未通过",topology_progress_label([item]))
        state["search_generation"]=4
        self.assertEqual(current_search_previews(state),[])


class PreviewWindowTests(unittest.TestCase):
    def setUp(self):
        self.window=window=object.__new__(AssistWindow)
        window.room=SimpleNamespace(name="a",x=0,y=0)
        window.offset=(0,0)
        window.scale=1
        window.shared=SharedAssistState()
        window.shared.update(mode="planning",search_previews=[entry(1),entry(2)])
        window.preview_styles=PreviewStyles()
        window.preview_filter_key=None
        window.preview_choices={"全部":None}
        window.preview_selector={}
        window.preview_choice=Value("全部")
        window.show_previews=Value(True)
        window.preview_last_draw=0
        window.preview_signature=None
        window.preview_tooltip=""
        window.preview_visible_workers=0
        window.strategy_labels={}
        window.canvas=Canvas()
        window.draft=RouteDraft()
        window.draft.append({"room":"a","x":8,"y":8,"tile_size":8})
        window.draw_dynamic=Mock()

    def test_preview_never_submits_commands_or_changes_user_draft(self):
        window=self.window
        before=deepcopy(window.draft.payload())
        window.draw_search_previews(force=True)
        self.assertEqual(window.draft.payload(),before)
        self.assertTrue(window.shared.commands.empty())
        self.assertEqual(window.preview_visible_workers,2)
        self.assertTrue(any(item[0]=="draft" for item in window.canvas.items))
        lines=[item for item in window.canvas.items if item[0]=="line"]
        self.assertEqual(len(lines),4)
        self.assertEqual(lines[0][1],(50,25,60,35))
        self.assertEqual(lines[1][1],(80,45,90,55))
        dots=[item for item in window.canvas.items if item[0]=="oval"]
        self.assertTrue(all(item[1]==(88,53,92,57) for item in dots))
        labels=[item[1] for item in window.canvas.items if item[0]=="text"]
        self.assertEqual(len(set(labels)),2)

    def test_local_leg_is_dashed_and_never_connected_to_player(self):
        window=self.window
        window.shared.update(search_previews=[entry(local=True)],observation={"room":"a","x":0,"y":0})
        window.draw_search_previews(force=True)
        lines=[item for item in window.canvas.items if item[0]=="line"]
        self.assertTrue(all(item[2]["dash"]==(2,5) for item in lines))
        self.assertEqual(lines[0][1][:2],(50,25))
        self.assertTrue(any("局部" in item[2].get("text","") for item in window.canvas.items if item[0]=="text"))
        self.assertIn("含局部片段",window.preview_tooltip)
        self.assertTrue(window.shared.commands.empty())

    def test_redraw_is_throttled_but_cancel_clears_immediately(self):
        window=self.window
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1):
            window.draw_search_previews()
        first=window.canvas.line_calls
        window.shared.update(search_previews=[entry(1,updated=2)])
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1.1):
            window.draw_search_previews()
        self.assertEqual(window.canvas.line_calls,first)
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1.5):
            window.draw_search_previews()
        self.assertGreater(window.canvas.line_calls,first)
        window.shared.update(mode="failed")
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1.51):
            window.draw_search_previews()
        self.assertEqual(window.preview_visible_workers,0)
        self.assertEqual(window.canvas.find_withtag("search_preview"),[])

    def test_worker_filter_and_toggle_only_affect_the_preview_layer(self):
        window=self.window
        window.draw_search_previews(force=True)
        selected=next(label for label,worker in window.preview_choices.items() if worker=="2")
        window.preview_choice.set(selected)
        window.preview_filter_changed()
        self.assertEqual(window.preview_visible_workers,1)
        color=window.preview_styles.get("2")[1]
        self.assertTrue(all(item[2]["fill"]==color for item in window.canvas.items if item[0]=="line"))
        window.show_previews.set(False)
        window.preview_filter_changed()
        self.assertEqual(window.canvas.find_withtag("search_preview"),[])
        self.assertTrue(window.shared.commands.empty())
        self.assertEqual(len(window.draft.waypoints),1)

    def test_coarse_corridors_are_gray_dotted_and_do_not_change_user_route(self):
        window=self.window
        before=deepcopy(window.draft.payload())
        window.shared.update(search_previews=[topology_entry()])
        window.draw_search_previews(force=True)
        lines=[item for item in window.canvas.items if item[0]=="line"]
        self.assertEqual(len(lines),3)
        self.assertTrue(all(item[2]["fill"]=="#94a3b8" and item[2]["dash"]==(1,5) for item in lines))
        self.assertEqual(lines[0][1],(10,25,40,45))
        self.assertIn("尚未尝试",window.preview_tooltip)
        self.assertIn("中转位置无法到达",window.preview_tooltip)
        self.assertIn("模拟已过 0 点",window.preview_tooltip)
        self.assertNotIn("整路已验证",window.preview_tooltip)
        self.assertIn("通路 2/3",window.preview_topology_status)
        self.assertEqual(window.draft.payload(),before)
        self.assertTrue(window.shared.commands.empty())

    def test_verified_geometry_remains_dotted_beside_real_simulator_paths(self):
        window=self.window
        item=topology_entry(physical=True)
        item["topology"]["corridors"][1]["status"]="verified"
        window.shared.update(search_previews=[item])
        window.draw_search_previews(force=True)
        lines=[item for item in window.canvas.items if item[0]=="line"]
        geometry=[item for item in lines if item[2]["fill"]=="#94a3b8"]
        physical=[item for item in lines if item[2]["fill"]!="#94a3b8"]
        self.assertEqual(len(geometry),3)
        self.assertTrue(all(item[2]["dash"]==(1,5) for item in geometry))
        self.assertEqual(len(physical),2)
        self.assertTrue(all("dash" not in item[2] for item in physical))
        self.assertIn("整路已验证",window.preview_topology_status)
        self.assertTrue(window.shared.commands.empty())

    def test_corridor_status_updates_redraw_and_cancel_clears_geometry(self):
        window=self.window
        item=topology_entry()
        window.shared.update(search_previews=[item])
        window.draw_search_previews(force=True)
        item=deepcopy(item)
        item["topology"]["corridors"][1].update(status="failed",reason="replay_failed")
        item["updated_at"]=2
        window.shared.update(search_previews=[item])
        window.draw_search_previews(force=True)
        self.assertIn("整路模拟未通过",window.preview_tooltip)
        self.assertIn("2 条未通过",window.preview_topology_status)
        window.shared.update(mode="manual")
        window.draw_search_previews(force=True)
        self.assertEqual(window.preview_topology_status,"")
        self.assertEqual(window.canvas.find_withtag("search_preview"),[])

    def test_explicit_empty_topology_clears_immediately_inside_redraw_throttle(self):
        window=self.window
        window.shared.update(search_previews=[topology_entry()])
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1):
            window.draw_search_previews()
        self.assertTrue(window.canvas.find_withtag("search_preview"))
        item=topology_entry()
        item["topology"]=dict(active_id=None,corridors=[])
        window.shared.update(search_previews=[item])
        with patch("celeste_search.assist_gui.time.monotonic",return_value=1.01):
            window.draw_search_previews()
        self.assertEqual(window.canvas.find_withtag("search_preview"),[])
        self.assertEqual(window.preview_topology_status,"")

    def test_old_generation_geometry_is_removed_without_touching_confirmed_draft(self):
        window=self.window
        before=deepcopy(window.draft.payload())
        window.shared.update(search_generation=3,search_previews=[topology_entry()])
        window.draw_search_previews(force=True)
        self.assertTrue(window.canvas.find_withtag("search_preview"))
        window.shared.update(search_generation=4)
        window.draw_search_previews(force=True)
        self.assertEqual(window.canvas.find_withtag("search_preview"),[])
        self.assertEqual(window.draft.payload(),before)
        self.assertTrue(window.shared.commands.empty())


if __name__=="__main__":
    unittest.main()
