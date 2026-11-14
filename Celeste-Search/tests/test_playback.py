import json
from pathlib import Path
import socket
import struct
import threading
import unittest
from types import SimpleNamespace

from celeste_search.playback import read_message, recv_exact, session_xml, distance_to_prediction


class PlaybackTests(unittest.TestCase):
    def test_fragmented_state_is_one_message(self):
        state={"schema_version":"celeste-playback-state-v1","room":"蔚蓝","frame":0}
        encoded=json.dumps(state,ensure_ascii=False).encode("utf-8")
        left,right=socket.socketpair()
        packet=b"STATE"+struct.pack("<I",len(encoded))+encoded
        def writer():
            with left:
                for byte in packet:
                    left.sendall(bytes([byte]))
        thread=threading.Thread(target=writer)
        thread.start()
        with right:
            self.assertEqual(read_message(right),("state",state))
        thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_disconnect_and_oversized_messages_fail(self):
        for payload in (b"ST",b"STATE"+struct.pack("<I",2_000_000)):
            left,right=socket.socketpair()
            left.sendall(payload)
            left.close()
            with right,self.assertRaises((ConnectionError,ValueError)):
                read_message(right)

    def test_native_chapter_and_side_handshake(self):
        for name,area,mode in (("1-ForsakenCity",1,0),("9H-Core",9,1),("LostLevels",10,0)):
            route={"map":{"path":name+".bin"},"start_room":"2", "predicted":{"state":{"x":1,"y":2}}}
            document=session_xml(route,30).decode()
            self.assertIn(f"<AreaKey>{area}</AreaKey>",document)
            self.assertIn(f"<AreaMode>{mode}</AreaMode>",document)
            self.assertIn("<SearchMode>true</SearchMode>",document)

    def test_drift_cannot_match_another_room(self):
        states=[SimpleNamespace(room="1",x=10,y=20),SimpleNamespace(room="1",x=14,y=20)]
        self.assertEqual(distance_to_prediction({"room":"1","x":10,"y":20},states,1),4)
        self.assertEqual(distance_to_prediction({"room":"2","x":10,"y":20},states,1),float("inf"))


if __name__=="__main__":
    unittest.main()
