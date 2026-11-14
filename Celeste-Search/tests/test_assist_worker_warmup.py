"""Explicit pool startup and service ordering; no real worker or game is used."""
from concurrent.futures import Future
import socket
from threading import Event
from unittest.mock import MagicMock,patch
import unittest

from celeste_search.assist import TargetPlanner,SharedAssistState,serve_assist,_warm_search_worker


class WarmPool:
    def __init__(self,mode="ready"):
        self.planner=None
        self.mode=mode
        self.calls=[]

    def submit(self,function,timeout):
        assert function is _warm_search_worker
        assert not self.planner.warm_release.is_set()
        future=Future()
        self.calls.append(future)
        if self.mode=="ready" or self.mode=="duplicate":
            self.planner.warm_ready.value+=1
            future.set_result(100 if self.mode=="duplicate" else 100+len(self.calls))
        elif self.mode=="error":
            future.set_exception(RuntimeError("worker initialization failed"))
        else:
            future.set_running_or_notify_cancel()
        return future

    def shutdown(self,**kwargs):
        pass


class WarmupTests(unittest.TestCase):
    def planner(self,mode="ready"):
        pool=WarmPool(mode)
        planner=TargetPlanner(workers=3,pool=pool)
        pool.planner=planner
        self.addCleanup(planner.close)
        return planner,pool

    def test_constructor_is_lazy_and_explicit_warmup_reaches_all_workers_once(self):
        planner,pool=self.planner()
        self.assertEqual(pool.calls,[])
        self.assertTrue(planner.warm_release.is_set())
        result=planner.warmup()
        self.assertEqual(result["ready"],3)
        self.assertEqual(result["worker_pids"],[101,102,103])
        self.assertFalse(result["cancelled"])
        self.assertEqual(planner.warmup(),result)
        self.assertEqual(len(pool.calls),3)
        self.assertEqual(planner.jobs,{})
        self.assertEqual(planner.stats()["submitted"],0)
        self.assertTrue(planner.warm_release.is_set())

    def test_barrier_timeout_releases_workers_and_does_not_cache_readiness(self):
        planner,pool=self.planner("stalled")
        with self.assertRaises(TimeoutError):
            planner.warmup(timeout=.01)
        self.assertTrue(planner.warm_release.is_set())
        self.assertIsNone(planner.warmup_result)

    def test_initialization_failure_releases_barrier_and_is_reported(self):
        planner,pool=self.planner("error")
        with self.assertRaisesRegex(RuntimeError,"initialization failed"):
            planner.warmup()
        self.assertTrue(planner.warm_release.is_set())
        self.assertIsNone(planner.warmup_result)

    def test_early_stop_does_not_submit_or_leave_a_blocking_barrier(self):
        planner,pool=self.planner()
        stop=Event()
        stop.set()
        self.assertTrue(planner.warmup(stop=stop)["cancelled"])
        self.assertEqual(pool.calls,[])
        self.assertTrue(planner.warm_release.is_set())

    def test_duplicate_fast_probes_do_not_prove_all_workers_ready(self):
        planner,pool=self.planner("duplicate")
        with self.assertRaisesRegex(RuntimeError,"every process"):
            planner.warmup()
        self.assertTrue(planner.warm_release.is_set())

    def test_warmup_cannot_block_an_active_search(self):
        planner,pool=self.planner()
        planner.active=True
        with self.assertRaisesRegex(RuntimeError,"precede search"):
            planner.warmup()
        self.assertEqual(pool.calls,[])


class ServiceStartupTests(unittest.TestCase):
    def test_listener_is_created_only_after_worker_warmup_completes(self):
        shared=SharedAssistState()
        order=[]
        planner=MagicMock()
        planner.workers=3
        planner.warmup.side_effect=lambda **kwargs:(order.append("warmup") or {"workers":3,"ready":3})
        listener=MagicMock()
        listener.__enter__.return_value=listener
        listener.listen.side_effect=lambda backlog:shared.stop.set()
        traces=[]
        with patch("celeste_search.assist.TargetPlanner",return_value=planner), \
             patch("celeste_search.assist.socket.socket",side_effect=lambda:(order.append("socket") or listener)):
            serve_assist(shared,"unused",25102,on_trace=traces.append)
        self.assertEqual(order,["warmup","socket"])
        self.assertEqual([row["kind"] for row in traces],
                         ["worker_warmup_started","worker_warmup_completed"])
        planner.close.assert_called_once()

    def test_failed_warmup_never_accepts_game_and_reports_startup_error(self):
        shared=SharedAssistState()
        planner=MagicMock()
        planner.workers=3
        planner.warmup.side_effect=TimeoutError("not all workers ready")
        traces=[]
        with patch("celeste_search.assist.TargetPlanner",return_value=planner), \
             patch("celeste_search.assist.socket.socket") as factory:
            serve_assist(shared,"unused",25102,on_trace=traces.append)
        factory.assert_not_called()
        self.assertEqual(shared.snapshot()["mode"],"failed")
        self.assertIn("not all workers ready",shared.snapshot()["message"])
        self.assertEqual(traces[-1]["kind"],"startup_error")
        self.assertEqual(traces[-1]["error_type"],"TimeoutError")

    def test_handshake_exception_is_kept_in_diagnostic_trace(self):
        shared=SharedAssistState()
        planner=MagicMock()
        planner.workers=3
        planner.warmup.return_value={"workers":3,"ready":3}
        listener=MagicMock()
        listener.__enter__.return_value=listener
        connection=MagicMock()
        accepted=[]

        def accept():
            if not accepted:
                accepted.append(True)
                return connection,("127.0.0.1",1234)
            shared.stop.set()
            raise socket.timeout()

        listener.accept.side_effect=accept
        traces=[]
        with patch("celeste_search.assist.TargetPlanner",return_value=planner), \
             patch("celeste_search.assist.socket.socket",return_value=listener), \
             patch("celeste_search.assist.recv_exact",return_value=b"BADHELLO"):
            serve_assist(shared,"unused",25102,on_trace=traces.append)
        error=next(row for row in traces if row["kind"]=="connection_error")
        self.assertEqual(error["error_type"],"ValueError")
        self.assertTrue(error["error"])
        self.assertEqual(traces[-1]["kind"],"disconnected")
        connection.close.assert_called_once()


if __name__=="__main__":
    unittest.main()
