"""Preview traffic is bounded, stale-safe, and never an executable candidate."""
from concurrent.futures import Future
import multiprocessing
import os
import pickle
from queue import Queue
import struct
from threading import Event, Thread
import time
import unittest

from celeste_search.assist import TargetPlanner


class Pool:
    def submit(self,*args):
        return Future()

    def shutdown(self,**kwargs):
        pass


class PreviewTransportTests(unittest.TestCase):
    def setUp(self):
        self.planner=TargetPlanner(workers=3,pool=Pool())
        self.queue=self.planner._progress_packets
        self.planner.specs=[dict(name="ordered_fine",family="ordered")]
        self.addCleanup(self.planner.close)

    def packet(self,**updates):
        packet=dict(generation=self.planner.epoch,worker_id=123,strategy="ordered_fine",
                    expansions=40,simulated_frames=200,waypoints_reached=1,
                    paths=[dict(kind="best",points=[dict(room="a",x=4,y=16),dict(room="b",x=8,y=12)],
                                local_leg=True,waypoints_reached=1)])
        packet.update(updates)
        return packet

    def test_preview_is_not_a_candidate_and_keeps_room_boundaries(self):
        self.queue.put(self.packet(actions=[32],can_execute=True))
        self.assertIsNone(self.planner.poll())
        entry=self.planner.previews()[0]
        self.assertNotIn("actions",entry)
        self.assertNotIn("can_execute",entry)
        self.assertTrue(entry["paths"][0]["local_leg"])
        self.assertEqual([p["room"] for p in entry["paths"][0]["points"]],["a","b"])
        self.assertEqual(self.planner.results,[])

    def test_cancellation_clears_visible_paths_and_drops_late_old_packets(self):
        old=self.packet()
        self.queue.put(old)
        self.planner.poll()
        self.assertEqual(len(self.planner.previews()),1)
        self.planner.cancel()
        self.queue.put(old)
        self.planner.poll()
        self.assertEqual(self.planner.previews(),[])

    def test_each_input_tick_drains_at_most_eight_packets(self):
        for worker in range(10):
            self.queue.put(self.packet(worker_id=worker))
        self.planner.poll()
        self.assertEqual(len(self.planner.previews()),8)
        self.assertEqual(self.queue.qsize(),2)

    def test_paths_and_points_are_bounded_and_invalid_coordinates_are_ignored(self):
        points=[dict(room="a",x=index,y=16) for index in range(200)]
        paths=[dict(kind="best",points=points) for _ in range(4)]
        self.queue.put(self.packet(paths=paths))
        self.planner.poll()
        accepted=self.planner.previews()[0]["paths"]
        self.assertEqual(len(accepted),2)
        self.assertEqual([len(path["points"]) for path in accepted],[120,120])
        self.queue.put(self.packet(worker_id=124,paths=[dict(points=[dict(room="a",x=float("nan"),y=0)])]))
        self.planner.poll()
        self.assertEqual(len(self.planner.previews()),1)

    def test_malformed_or_unknown_strategy_packets_do_not_interrupt_polling(self):
        for packet in (None,self.packet(paths=4),self.packet(paths=[dict(points=5)]),
                       self.packet(strategy="unknown"),self.packet(worker_id="123")):
            self.queue.put(packet)
        self.queue.put(self.packet())
        self.assertIsNone(self.planner.poll())
        self.assertEqual(len(self.planner.previews()),1)

    def test_topology_only_is_visible_but_never_executable_or_waypoint_progress(self):
        corridor=dict(id="route-a",status="trying",reason="验证下一通道",
                      points=[dict(room="a",x=4,y=16),dict(room="a",x=12,y=24)],
                      actions=[32],can_execute=True,waypoints_reached=99)
        self.queue.put(self.packet(paths=[],waypoints_reached=0,
            topology=dict(active_id="route-a",corridors=[corridor])))
        self.assertIsNone(self.planner.poll())
        preview=self.planner.previews()[0]
        self.assertEqual(preview["paths"],[])
        self.assertEqual(preview["waypoints_reached"],0)
        self.assertEqual(set(preview["topology"]["corridors"][0]),{"id","status","reason","points"})
        self.assertEqual(self.planner.results,[])

    def test_topology_bounds_and_invalid_entries_cannot_escape_transport(self):
        rows=[dict(id=f"r-{index}",status="pending",reason="x"*400,
                   points=[dict(room="a",x=x,y=16) for x in range(100)]) for index in range(30)]
        self.queue.put(self.packet(paths=[],topology=dict(active_id="missing",corridors=rows)))
        self.planner.poll()
        topology=self.planner.previews()[0]["topology"]
        self.assertEqual(len(topology["corridors"]),24)
        self.assertIsNone(topology["active_id"])
        self.assertTrue(all(len(row["points"])==60 and len(row["reason"])==160 for row in topology["corridors"]))
        invalid=[dict(id="a",status="executing",points=rows[0]["points"]),
                 dict(id="b",status="trying",points=[dict(room="a",x=float("inf"),y=1)]),None]
        self.queue.put(self.packet(worker_id=124,paths=[],topology=dict(corridors=invalid)))
        self.planner.poll()
        self.assertEqual(len(self.planner.previews()),1)

    def test_topology_late_generation_is_discarded_with_physical_previews(self):
        packet=self.packet(paths=[],topology=dict(active_id=None,corridors=[
            dict(id="r",status="pending",points=[dict(room="a",x=4,y=16),dict(room="a",x=8,y=20)])]))
        self.queue.put(packet)
        self.planner.poll()
        self.assertTrue(self.planner.previews())
        self.planner.cancel()
        self.queue.put(packet)
        self.planner.poll()
        self.assertEqual(self.planner.previews(),[])

    def test_empty_topology_clears_old_lines_and_bad_knot_rejects_entire_corridor(self):
        points=[dict(room="a",x=4,y=16),dict(room="a",x=8,y=20)]
        self.queue.put(self.packet(paths=[],topology=dict(active_id=None,corridors=[
            dict(id="r",status="pending",points=points)])))
        self.planner.poll()
        self.queue.put(self.packet(paths=[],topology=dict(active_id=None,corridors=[])))
        self.planner.poll()
        self.assertEqual(self.planner.previews()[0]["topology"]["corridors"],[])
        broken=[points[0],dict(room="a",x=float("nan"),y=18),points[1]]
        self.queue.put(self.packet(worker_id=124,paths=[],topology=dict(active_id=None,corridors=[
            dict(id="r",status="pending",points=broken)])))
        self.planner.poll()
        self.assertEqual(len(self.planner.previews()),1)

    def test_reader_keeps_completed_transport_bounded_and_latest(self):
        class Source(Queue):
            def get(self,*args,**kwargs):
                value=super().get(*args,**kwargs)
                if value is marker:
                    drained.set()
                return value
        source=Source()
        packets=Queue(maxsize=3)
        stop,drained=Event(),Event()
        marker=object()
        for value in range(20):
            source.put(value)
        source.put(marker)
        reader=Thread(target=TargetPlanner._read_progress,args=(source,packets,stop),daemon=True)
        reader.start()
        try:
            self.assertTrue(drained.wait(1))
            stop.set()
            reader.join(timeout=.5)
            self.assertFalse(reader.is_alive())
            self.assertLessEqual(packets.qsize(),3)
            values=[]
            while not packets.empty():
                values.append(packets.get_nowait())
            # The marker wakes the test before enqueue, so stop may discard it;
            # all preceding completed values must still retain the latest ones.
            self.assertIn(19,values)
            self.assertNotIn(0,values)
        finally:
            stop.set()
            reader.join(timeout=.5)

    @unittest.skipIf(os.name=="nt","Raw POSIX pipe framing probe")
    def test_stdlib_get_nowait_waits_for_a_half_written_message(self):
        queue=multiprocessing.get_context("spawn").Queue(maxsize=1)
        payload=pickle.dumps({"probe":"complete"})
        # Match Queue.put's semaphore reservation, then deliberately publish a
        # valid frame header and only one body byte before the reader starts.
        self.assertTrue(queue._sem.acquire(False))
        os.write(queue._writer.fileno(),struct.pack("!i",len(payload))+payload[:1])
        started,done=Event(),Event()
        result=[]
        def read():
            started.set()
            result.append(queue.get_nowait())
            done.set()
        reader=Thread(target=read,daemon=True)
        reader.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertFalse(done.wait(.08),"get_nowait must expose the partial-pipe regression")
            os.write(queue._writer.fileno(),payload[1:])
            self.assertTrue(done.wait(1))
            self.assertEqual(result,[{"probe":"complete"}])
        finally:
            reader.join(timeout=1)
            queue.close()

    @unittest.skipIf(os.name=="nt","Raw POSIX pipe framing probe")
    def test_partial_ipc_packet_never_blocks_poll_and_late_generation_is_dropped(self):
        queue=self.planner.progress_queue
        packet=self.packet()
        payload=pickle.dumps(packet)
        self.assertTrue(queue._sem.acquire(False))
        os.write(queue._writer.fileno(),struct.pack("!i",len(payload))+payload[:1])
        # Reader can be inside recv_bytes indefinitely; poll must only read
        # the local complete-packet queue and return without producer help.
        started=time.monotonic()
        for _ in range(20):
            self.assertIsNone(self.planner.poll())
        self.assertLess(time.monotonic()-started,.04)
        self.assertEqual(self.planner.previews(),[])
        self.planner.cancel()
        os.write(queue._writer.fileno(),payload[1:])
        deadline=time.monotonic()+1
        while self.queue.empty() and time.monotonic()<deadline:
            time.sleep(.001)
        self.assertFalse(self.queue.empty())
        self.planner.poll()
        self.assertEqual(self.planner.previews(),[])

    @unittest.skipIf(os.name=="nt","Raw POSIX pipe framing probe")
    def test_close_does_not_join_a_reader_waiting_for_a_partial_packet(self):
        queue=self.planner.progress_queue
        payload=pickle.dumps(self.packet())
        self.assertTrue(queue._sem.acquire(False))
        os.write(queue._writer.fileno(),struct.pack("!i",len(payload))+payload[:1])
        deadline=time.monotonic()+1
        # Confirm the daemon owns the IPC read lock, proving it started the
        # incomplete read rather than merely being about to enter Queue.get.
        entered=False
        while time.monotonic()<deadline:
            if not queue._rlock.acquire(False):
                entered=True
                break
            queue._rlock.release()
            time.sleep(.001)
        self.assertTrue(entered)
        started=time.monotonic()
        self.planner.close()
        self.assertLess(time.monotonic()-started,.04)
        self.assertTrue(self.planner._progress_reader.daemon)
        os.write(queue._writer.fileno(),payload[1:])
        self.planner._progress_reader.join(timeout=1)
        self.assertFalse(self.planner._progress_reader.is_alive())
        self.assertTrue(self.queue.empty())


if __name__=="__main__":
    unittest.main()
