"""Test draft editing and atomic confirmation without a game or display server."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from celeste_search.assist import SharedAssistState
from celeste_search.assist_gui import AssistWindow,RouteDraft,search_display
from celeste_search.maps import MapData,Room


def target(x,room="a"):
    return {"room":room,"x":x,"y":16,"tile_size":8}


class RouteDraftTests(unittest.TestCase):
    def test_each_clicked_point_has_independent_incoming_constraints(self):
        draft=RouteDraft()
        draft.append(target(16))
        draft.set_constraints(True,False)
        draft.append(target(32))
        draft.set_constraints(False,True)
        self.assertEqual(draft.segments,[{"no_dash":True,"no_ground":False},
                                        {"no_dash":False,"no_ground":True}])

    def test_submitted_route_cannot_change_when_user_edits_draft(self):
        draft=RouteDraft()
        original=target(16)
        draft.append(original)
        draft.set_constraints(True,True)
        submitted=draft.payload()
        original["x"]=80
        draft.set_constraints(False,False)
        draft.append(target(32))
        draft.clear()
        self.assertEqual(submitted,{"waypoints":[target(16)],
                                    "segments":[{"no_dash":True,"no_ground":True}]})

    def test_reorder_keeps_constraints_with_incoming_destination(self):
        draft=RouteDraft()
        draft.append(target(16))
        draft.append(target(32,"b"))
        draft.set_constraints(True,True)
        draft.move(-1)
        self.assertEqual(draft.selected,0)
        self.assertEqual(draft.waypoints,[target(32,"b"),target(16)])
        self.assertEqual(draft.segments[0],{"no_dash":True,"no_ground":True})
        draft.move(-1)
        self.assertEqual(draft.selected,0)

    def test_removal_does_not_leave_an_unpaired_segment(self):
        draft=RouteDraft()
        for x in (16,32,48):
            draft.append(target(x))
        draft.selected=1
        draft.remove()
        self.assertEqual(draft.waypoints,[target(16),target(48)])
        self.assertEqual(len(draft.segments),2)
        draft.remove()
        draft.remove()
        draft.remove()
        self.assertEqual(draft.selected,-1)
        self.assertEqual(draft.payload(),{"waypoints":[],"segments":[]})


class SearchDisplayTests(unittest.TestCase):
    def test_counts_and_active_names_use_strategy_labels(self):
        stats={"active":2,"completed":7,"total":12,"strategies":["one","two","three"],
               "active_strategies":["two","three"]}
        counter,preview,complete=search_display(stats,{"one":"直达","two":"补充","three":"落台"})
        self.assertIn("2 运行 / 7 完成 / 12 总计",counter)
        self.assertIn("正在计算：补充 / 落台",preview)
        self.assertIn("1. 直达",complete)
        self.assertIn("3. 落台",complete)

    def test_long_portfolio_preview_is_bounded_but_details_keep_every_name(self):
        names=[f"strategy-{index}" for index in range(20)]
        counter,preview,complete=search_display({"active":0,"completed":0,"total":20,"strategies":names},{})
        self.assertIn("0 运行 / 0 完成 / 20 总计",counter)
        self.assertIn("等 20 种",preview)
        self.assertNotIn("strategy-19",preview)
        self.assertIn("20. strategy-19",complete)

    def test_no_portfolio_does_not_add_placeholder_counts(self):
        for stats in (None,{}, {"total":0}):
            self.assertEqual(search_display(stats,{}),("","",""))


class RouteWindowWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.window=window=object.__new__(AssistWindow)
        room=Room("a",0,0,64,64,("0"*8,)*8,(),((16.,48.),))
        window.map=MapData("fixture",{"a":room},Path("fixture.bin"))
        window.room=room
        window.draft=RouteDraft()
        window.scale=2
        window.offset=(20,20)
        window.shared=SharedAssistState()
        window.shared.update(connected=True)
        window.selection=SimpleNamespace(set=Mock())
        window.status=SimpleNamespace(set=Mock())
        window.canvas=SimpleNamespace(canvasx=lambda x:x,canvasy=lambda y:y)
        window.draw_dynamic=Mock()
        window.refresh_editor=Mock()

    def click(self,x,y=16):
        self.window.select_tile(SimpleNamespace(x=20+x*2,y=20+y*2))

    def test_several_clicks_are_only_a_draft_until_one_confirmation(self):
        window=self.window
        self.click(16)
        window.draft.set_constraints(True,False)
        self.click(32)
        self.assertTrue(window.shared.commands.empty())
        window.confirm()
        command=window.shared.commands.get_nowait()
        self.assertEqual(command["kind"],"route")
        self.assertEqual(command["route"]["waypoints"],[target(16),target(32)])
        self.assertTrue(command["route"]["segments"][0]["no_dash"])
        self.assertTrue(window.shared.commands.empty())
        window.draft.clear()
        self.assertEqual(len(command["route"]["waypoints"]),2)

    def test_invalid_click_does_not_destroy_existing_route(self):
        self.click(16)
        self.click(80)
        self.assertEqual(self.window.draft.waypoints,[target(16)])
        self.assertTrue(self.window.shared.commands.empty())

    def test_disconnected_or_empty_draft_cannot_confirm(self):
        self.window.confirm()
        self.click(16)
        self.window.shared.update(connected=False)
        self.window.confirm()
        self.assertTrue(self.window.shared.commands.empty())


if __name__=="__main__":
    unittest.main()
