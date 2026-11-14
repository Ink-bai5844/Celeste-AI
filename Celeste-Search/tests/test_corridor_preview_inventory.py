"""Later corridor pages survive publication, transport and native GUI data."""
import unittest
from unittest.mock import patch

from celeste_search.assist import _topology_preview
from celeste_search.assist_gui import (preview_corridors, corridor_caption,
                                     topology_progress_label,corridor_label_indices)
from celeste_search.physics import SimState
from celeste_search.search_progress import progress_reporter, publish_corridor_progress


class CorridorPreviewInventoryTests(unittest.TestCase):
    def test_rotation_throttles_transport_but_keeps_latest_inventory(self):
        from celeste_search.search_progress import _ACTIVE
        packets=[]
        start=SimState("a",16,48,on_ground=True)
        row=dict(id="route",status="pending",turns=0,
                 points=[dict(room="a",x=16,y=48),dict(room="a",x=80,y=40)])
        with progress_reporter(packets.append,start,dict(waypoints=[dict(room="a",x=80,y=40)])):
            with patch("celeste_search.search_progress.time.monotonic",return_value=10):
                publish_corridor_progress([row],throttle_seconds=.1)
                row.update(turns=2,status="trying")
                publish_corridor_progress([row],"route",throttle_seconds=.1)
                self.assertEqual(len(packets),1)
                self.assertEqual(_ACTIVE.get().topology["corridors"][0]["turns"],2)
                publish_corridor_progress([row],"route")
                self.assertEqual(packets[-1]["topology"]["corridors"][0]["turns"],2)

    def publish(self, rows, active):
        packets = []
        start = SimState("a",16,48,on_ground=True)
        route = dict(waypoints=[dict(room="a",x=80,y=40)],segments=[{}])
        with progress_reporter(packets.append,start,route):
            publish_corridor_progress(rows,active)
        return packets[-1]

    def test_active_corridor_on_fourth_page_is_visible_with_its_work_counts(self):
        rows = [dict(id=f"route-{i}",status="trying" if i==20 else "pending",
            turns=3 if i==20 else 1 if i<12 else 0,expansions=192 if i==20 else 0,
            actions=[32],can_execute=True,
            points=[dict(room="a",x=16,y=48),dict(room="a",x=80,y=40)])
            for i in range(24)]
        packet = self.publish(rows,"route-20")
        self.assertEqual(packet["waypoints_reached"],0)
        topology = _topology_preview(packet["topology"])
        visible = preview_corridors(dict(topology=topology))
        self.assertEqual(len(visible),24)
        self.assertEqual(topology["active_id"],"route-20")
        self.assertEqual(visible[20]["expansions"],192)
        self.assertEqual(visible[20]["turns"],3)
        self.assertNotIn("actions",str(topology))
        self.assertNotIn("can_execute",str(topology))
        label = topology_progress_label([dict(topology=topology)])
        self.assertIn("通路 21/24",label)
        self.assertIn("第 3 轮",label)
        self.assertIn("已尝试 13/24 条",label)
        labels=corridor_label_indices(visible,"route-20")
        self.assertIn(20,labels)
        self.assertEqual(len(labels),6)
        self.assertEqual(len(visible),24)

    def test_untried_and_suspended_routes_have_different_labels(self):
        row = dict(id="route",status="pending",reason="",points=[])
        self.assertIn("尚未尝试",corridor_caption(row,1,2))
        self.assertIn("等待下一轮",corridor_caption(dict(row,turns=2),1,2))
        self.assertIn("第 2 轮",corridor_caption(dict(row,turns=2),1,2))

    def test_work_counts_are_bounded_numbers_not_commands_or_untrusted_text(self):
        row = dict(id="route",status="pending",turns="execute",expansions=-10,
                   points=[dict(room="a",x=1,y=2),dict(room="a",x=3,y=4)])
        packet = self.publish([row],"route")
        self.assertNotIn("turns",packet["topology"]["corridors"][0])
        topology = _topology_preview(dict(active_id="route",corridors=[row]))
        self.assertNotIn("turns",topology["corridors"][0])
        self.assertNotIn("expansions",topology["corridors"][0])
        self.assertIn("尚未尝试",corridor_caption(row,1,1))


if __name__ == "__main__":
    unittest.main()
