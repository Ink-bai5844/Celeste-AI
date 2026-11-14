import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest

from celeste_search.exploration import Exploration
from celeste_search.live import accept_handshake
from celeste_search.maps import MapData, Room
from celeste_search.telemetry import Telemetry


def handshake_fixture():
    # The handshake only needs a playable start room and the chapter filename.
    # Keep the real chapter identity without reading owned game assets.
    source = Path("synthetic-maps") / "1-ForsakenCity.bin"
    room = Room("start", 0, 0, 64, 64, ("0" * 8,) * 7 + ("1" * 8,),
                (), ((16.0, 56.0),))
    return MapData("1-ForsakenCity", {room.name: room}, source)


class PersistenceTests(unittest.TestCase):
    def test_only_actual_crossings_count_as_explored_edges(self):
        ledger=Exploration()
        for room,spawn,x in (("a",1,0),("a",1,32),("b",1,320),("a",2,0)):
            ledger.observe(dict(room=room,spawn_id=spawn,x=x,y=0,frame=0))
        saved=ledger.snapshot()
        self.assertEqual(saved["edge_visits"],{"a>b":1})
        self.assertEqual(saved["room_visits"],{"a":2,"b":1})
        self.assertEqual(len(saved["visited_cells"]),3)
        restored=Exploration(saved)
        self.assertEqual(restored.snapshot(),saved)

    def test_writer_persists_execution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            logger=Telemetry(Path(directory))
            logger.emit("frame",{"frame":1,"mask":16})
            logger.emit("event",{"event":"berry_confirmed","berries":["2:11"]})
            logger.emit("plan",{"run_plan_id":1,"actions":[16,0]})
            logger.close()
            self.assertIsNone(logger.error)
            self.assertEqual(logger.dropped,0)
            self.assertEqual(json.loads((Path(directory)/"frames.jsonl").read_text()),{"frame":1,"mask":16})
            self.assertEqual(json.loads((Path(directory)/"plan-00001.json").read_text())["actions"],[16,0])

    def test_opened_walls_require_actual_runtime_removal(self):
        ledger=Exploration()
        observation=dict(room="a",spawn_id=1,x=0,y=0,frame=0,
                         room_entities={"entities":[dict(kind="dashBlock",id="a:3",present=True)],
                                        "do_not_load":[]})
        ledger.observe(observation)
        self.assertEqual(ledger.snapshot()["opened_walls"],[])
        observation["room_entities"]["do_not_load"]=["a:3"]
        ledger.observe(observation)
        self.assertEqual(ledger.snapshot()["opened_walls"],["a:3"])

    def test_live_handshake_accepts_current_chapter_without_loading(self):
        data = handshake_fixture()
        controller,bridge=socket.socketpair()
        self.addCleanup(controller.close)
        self.addCleanup(bridge.close)
        hello=json.dumps({"area_id":1,"area_mode":0,"frame":15000}).encode()
        bridge.sendall(b"LIVEINFO"+struct.pack("<I",len(hello))+hello)
        self.assertTrue(accept_handshake(controller,data,data.source))
        self.assertEqual(bridge.recv(32),b"\x00")

    def test_live_handshake_refuses_different_chapter_without_loading(self):
        data = handshake_fixture()
        controller,bridge=socket.socketpair()
        self.addCleanup(controller.close)
        self.addCleanup(bridge.close)
        hello=json.dumps({"area_id":2,"area_mode":0}).encode()
        bridge.sendall(b"LIVEINFO"+struct.pack("<I",len(hello))+hello)
        with self.assertRaisesRegex(ValueError,"differs"):
            accept_handshake(controller,data,data.source)
        bridge.settimeout(.02)
        with self.assertRaises(TimeoutError):
            bridge.recv(1)

    def test_reconnect_cannot_send_a_new_chapter_load(self):
        data = handshake_fixture()
        controller,bridge=socket.socketpair()
        self.addCleanup(controller.close)
        self.addCleanup(bridge.close)
        bridge.sendall(b"LOADINFO")
        with self.assertRaisesRegex(ValueError,"preserve"):
            accept_handshake(controller,data,data.source,allow_new_session=False)
        bridge.settimeout(.02)
        with self.assertRaises(TimeoutError):
            bridge.recv(1)
