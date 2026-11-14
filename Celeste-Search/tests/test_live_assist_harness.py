"""Developer inbox/trace checks; no native window or game is launched."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from celeste_search.assist import SharedAssistState
from celeste_search.maps import MapData,Room

source=Path(__file__).resolve().parents[2]/"tools"/"live_assist_harness.py"
spec=importlib.util.spec_from_file_location("live_assist_harness",source)
harness=importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)
        self.shared=SharedAssistState()
        room=Room("a",0,0,64,64,("0"*8,)*8,(),((16.,48.),))
        self.shared.update(connected=True,request_id=3,map_data=MapData("fixture",{"a":room},Path("fixture.bin")),
                           observation={"room":"a","spawn_id":4})
        self.command={"id":"one","kind":"route","expected_request_id":3,"expected_room":"a","expected_spawn_id":4,
                      "route":{"waypoints":[{"room":"a","x":32,"y":16,"tile_size":8}],
                               "segments":[{"no_dash":True,"no_ground":False}]}}
        self.inbox=harness.CommandInbox(self.folder/"command.json",self.folder/"ack.json",self.shared)

    def submit(self,command=None):
        harness.write_json(self.folder/"command.json",command or self.command)
        return self.inbox.poll()

    def test_valid_explicit_command_preserves_route_and_execution_guard(self):
        ack=self.submit()
        self.assertEqual(ack["status"],"queued")
        self.assertEqual(ack["submitted_request_id"],4)
        queued=self.shared.commands.get_nowait()
        self.assertEqual(queued["route"],self.command["route"])
        self.assertEqual(queued["guard"],{"request_id":3,"room":"a","spawn_id":4})
        self.command["route"]["waypoints"][0]["x"]=48
        self.assertEqual(queued["route"]["waypoints"][0]["x"],32)
        self.assertIsNone(self.inbox.poll())
        self.assertTrue(self.shared.commands.empty())

    def test_context_changes_reject_without_queueing(self):
        for field,value,reason in (("expected_request_id",2,"request_changed"),
                                   ("expected_spawn_id",3,"spawn_changed"),
                                   ("expected_room","b","room_changed")):
            self.command.update(id=field)
            self.command[field]=value
            self.assertEqual(self.submit()["reason"],reason)
            self.command.update(expected_request_id=3,expected_spawn_id=4,expected_room="a")
        self.assertTrue(self.shared.commands.empty())

    def test_disconnected_and_missing_request_context_rejected(self):
        self.shared.update(connected=False)
        self.assertEqual(self.submit()["reason"],"game_not_connected")
        self.shared.update(connected=True)
        self.command["id"]="two"
        del self.command["expected_request_id"]
        self.assertEqual(self.submit()["reason"],"request_changed")
        self.assertTrue(self.shared.commands.empty())

    def test_startup_file_and_reused_id_never_replay(self):
        harness.write_json(self.folder/"command.json",self.command)
        restarted=harness.CommandInbox(self.folder/"command.json",self.folder/"ack.json",self.shared)
        self.assertIsNone(restarted.poll())
        self.command["route"]["waypoints"][0]["x"]=48
        harness.write_json(self.folder/"command.json",self.command)
        self.assertEqual(restarted.poll()["reason"],"duplicate_command_id")
        self.assertTrue(self.shared.commands.empty())

    def test_invalid_route_and_unknown_command_do_not_submit(self):
        self.command["route"]["segments"][0]["no_dash"]="yes"
        self.assertEqual(self.submit()["status"],"rejected")
        self.command.update(id="two",kind="restart")
        self.assertEqual(self.submit()["reason"],"unsupported_command")
        self.assertTrue(self.shared.commands.empty())

    def test_explicit_cancel_uses_same_context_guard(self):
        self.command.update(kind="cancel")
        ack=self.submit()
        self.assertEqual(ack["submitted_request_id"],3)
        queued=self.shared.commands.get_nowait()
        self.assertEqual(queued["kind"],"cancel")
        self.assertNotIn("route",queued)


class TraceTests(unittest.TestCase):
    def test_writer_drains_records_and_reports_any_bounded_queue_loss(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"trace.jsonl"
            writer=harness.TraceWriter(path,max_pending=2)
            for frame in range(100):
                writer.enqueue({"kind":"state","observation":{"frame":frame},"action":128})
            stats=writer.close()
            self.assertIsNone(stats["error"])
            self.assertEqual(stats["written"]+stats["dropped"],100)
            rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows),stats["written"]+1)
            self.assertEqual(rows[-1]["kind"],"trace_closed")
            self.assertEqual(rows[-1]["dropped"],stats["dropped"])

    def test_passive_sampling_preserves_full_active_states_and_execution_tail(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"trace.jsonl"
            clock=[0.0]
            writer=harness.TraceWriter(path,clock=lambda:clock[0])
            expected=[]

            def emit(at,mode,label,keep=True,**changes):
                clock[0]=at
                record={"kind":"state","mode":mode,"request_id":1,"label":label,
                        "observation":{"room":"a","spawn_id":1,"dead":False}}
                record.update(changes)
                writer.enqueue(record)
                if keep:
                    expected.append(label)

            emit(0,"manual","first")
            emit(.1,"manual","sample-out",False)
            emit(1,"manual","one-second")
            emit(1.01,"planning","planning-1")
            emit(1.02,"planning","planning-2")
            emit(1.03,"waiting","waiting-1")
            emit(1.04,"waiting","waiting-2")
            emit(2,"executing","executing-1")
            emit(2.01,"executing","executing-2")
            emit(2.02,"failed","terminal")
            emit(2.03,"failed","tail-1")
            emit(3,"failed","tail-2")
            emit(4,"failed","tail-end")
            emit(4.03,"failed","tail-expired",False)
            emit(5,"failed","passive-again")
            emit(5.01,"success","terminal-change")
            emit(5.02,"manual","manual-change")
            emit(5.03,"manual","new-request",request_id=2)
            emit(5.04,"manual","new-spawn",request_id=2,
                 observation={"room":"a","spawn_id":2,"dead":False})
            emit(5.05,"manual","new-room",request_id=2,
                 observation={"room":"b","spawn_id":2,"dead":False})
            emit(5.06,"manual","death",request_id=2,
                 observation={"room":"b","spawn_id":2,"dead":True})
            writer.enqueue({"kind":"event","event":{"name":"always-retained"}})
            stats=writer.close()
            rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["label"] for row in rows if row["kind"]=="state"],expected)
            self.assertEqual(stats["sampled_out"],2)
            self.assertEqual(stats["dropped"],0)
            self.assertTrue(any(row["kind"]=="event" for row in rows))
            self.assertEqual(rows[-1]["sampled_out"],2)


if __name__=="__main__":
    unittest.main()
