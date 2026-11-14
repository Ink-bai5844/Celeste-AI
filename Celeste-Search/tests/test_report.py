from pathlib import Path
import json
import tempfile
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.report import write_report


class ReportTests(unittest.TestCase):
    def test_report_is_offline_escaped_and_explicitly_predicted(self):
        name = '1</script><script>alert("room")</script>'
        room = Room(name, 0, 0, 16, 16, ("11", "00"),
                    (Entity("strawberry", 1, 8, 8, 0, 0),), ((4, 8),))
        data = MapData('<script>alert("package")</script>', {name: room}, Path("fake.bin"))
        route = {"start_room": name, "frames": 12,
                 "predicted": {"berries": [name+":1"], "rooms": [name], "coverage_cells": [[name, 0, 0]]},
                 "trace": [{"frame": 0, "room": name, "x": 4, "y": 8},
                           {"frame": 12, "room": name, "x": 8, "y": 8}],
                 "search": {"legs": [{"goal": "<img onerror=alert(1)>", "reached": True}]},
                 "sources": ["javascript:alert(1)", "https://example.org/reference?a=1&b=2"],
                 "runtime_verified": False}
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(data, route, Path(directory) / "report.html")
            html = path.read_text(encoding="utf-8")
        self.assertIn("仅模拟预测 · 尚未实机验证", html)
        self.assertIn("1 / 1", html)
        self.assertIn('class="berry predicted"', html)
        self.assertIn('class="route-path"', html)
        self.assertNotIn('<script>alert(', html)
        self.assertNotIn('<img onerror=', html)
        self.assertNotIn('href="javascript:', html)
        self.assertNotIn('src="http', html)
        self.assertNotIn("fetch(", html)
        self.assertNotIn("@@", html)
        payload = html.split('<script id="map-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(payload)["start"], name)

    def test_minimal_route_and_unknown_capabilities_are_renderable(self):
        room = Room("1", -8, -8, 16, 16, ("00", "11"), (), ((0, 0),))
        data = MapData("test", {"1": room}, Path("test.bin"))
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(data, {}, Path(directory) / "nested" / "report.html")
            html = path.read_text(encoding="utf-8")
        self.assertIn("暂无搜索阶段记录", html)
        self.assertIn("0 / 1", html)
        self.assertIn("M-8 0h16v8h-16z", html)
        self.assertNotIn("@@", html)


if __name__ == "__main__":
    unittest.main()
