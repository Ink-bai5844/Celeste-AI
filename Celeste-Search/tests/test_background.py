"""Nonblocking scheduling tests with manually completed futures; no game."""

from concurrent.futures import Future
from unittest.mock import patch
import unittest

from celeste_search.background import BackgroundRoomPlanner
from celeste_search.physics import SimState


class CheckedFuture(Future):
    def __init__(self):
        super().__init__()
        self.result_calls = 0
        self.set_running_or_notify_cancel()

    def result(self, timeout=None):
        if not self.done():
            raise AssertionError("game thread attempted to wait for unfinished work")
        self.result_calls += 1
        return super().result(timeout=timeout)


class FakeExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submissions = []
        self.shutdown_calls = []

    def submit(self, function, config):
        future = CheckedFuture()
        self.submissions.append((future, function, config))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def candidate(*, room="a", berries=(), distance=24, mask=1):
    return {"can_execute":True, "actions":[mask]*12, "status":"transition" if room!="a" else "partial",
            "final_state":{"room":room,"berries":list(berries),"x":24+distance,"y":64}}


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        patcher=patch("celeste_search.background.ProcessPoolExecutor",FakeExecutor)
        patcher.start();self.addCleanup(patcher.stop)
        self.planner=BackgroundRoomPlanner("fixture.bin",workers=2,seconds=2,expansions=200)
        self.addCleanup(self.planner.close)
        self.executor=self.planner._executor
        self.state=SimState("a",24,64,on_ground=True)

    def test_request_bounds_work_and_poll_never_waits(self):
        self.assertTrue(self.planner.request(self.state,4,["rejected"],["a"]))
        self.assertTrue(self.planner.busy)
        self.assertFalse(self.planner.request(self.state,6,[],[]))
        self.assertEqual(len(self.executor.submissions),2)
        self.assertIsNone(self.planner.poll())
        self.assertTrue(all(f.result_calls==0 for f,_,_ in self.executor.submissions))
        configs=[c for _,_,c in self.executor.submissions]
        self.assertEqual([c["variant"] for c in configs],[4,5])
        self.assertEqual(configs[0]["forbidden"],["rejected"])
        self.assertEqual(configs[0]["seconds"],2)

    def test_partial_waits_for_other_worker_then_selects_best(self):
        self.planner.request(self.state)
        first,second=[row[0] for row in self.executor.submissions]
        first.set_result(candidate(distance=24))
        self.assertIsNone(self.planner.poll())
        self.assertEqual(second.result_calls,0)
        second.set_result(candidate(distance=48,mask=2))
        result=self.planner.poll()
        self.assertEqual(result["actions"],[2]*12)
        self.assertEqual(result["anchor_state"]["x"],24)
        self.assertEqual(result["generation"],self.planner.generation)
        self.assertIsNone(self.planner.poll(),"a generation must publish at most once")
        self.assertFalse(self.planner.busy)

    def test_early_good_result_leaves_stale_worker_capacity_bounded(self):
        self.planner.request(self.state)
        first,stale=[row[0] for row in self.executor.submissions]
        first.set_result(candidate(room="b"))
        result=self.planner.poll()
        self.assertEqual(result["final_state"]["room"],"b")
        self.assertEqual(stale.result_calls,0)
        self.assertTrue(self.planner.busy)
        next_state=SimState("b",344,64,on_ground=True)
        self.assertTrue(self.planner.request(next_state,10))
        self.assertEqual(len(self.executor.submissions),3,"only the freed one of two slots is reusable")
        self.assertFalse(self.planner.request(next_state,20))
        stale.set_result(candidate(berries=["a:1","a:2"]))
        self.assertIsNone(self.planner.poll(),"an old anchor must never publish into the next generation")
        self.assertEqual(stale.result_calls,0)
        current=self.executor.submissions[-1][0]
        current.set_result(candidate(room="b",distance=48))
        result=self.planner.poll()
        self.assertEqual(result["anchor_state"]["room"],"b")
        self.assertGreater(result["generation"],1)

    def test_cancel_invalidates_running_results_without_freeing_fake_capacity(self):
        self.planner.request(self.state)
        before=self.planner.generation
        self.planner.cancel_epoch()
        self.assertGreater(self.planner.generation,before)
        self.assertFalse(self.planner.request(self.state),"running stale jobs still occupy their slots")
        for future,_,_ in self.executor.submissions:
            future.set_result(candidate(room="b"))
        self.assertIsNone(self.planner.poll())
        self.assertTrue(self.planner.request(self.state))
        self.assertEqual(len(self.executor.submissions),4)

    def test_berry_gain_publishes_early_and_worker_error_is_inspectable(self):
        self.planner.request(self.state)
        first,second=[row[0] for row in self.executor.submissions]
        first.set_result(candidate(berries=["a:7"]))
        result=self.planner.poll()
        self.assertEqual(result["final_state"]["berries"],["a:7"])
        second.set_result(candidate())
        self.planner.poll()
        self.planner.request(self.state)
        for future,_,_ in self.executor.submissions[2:]:
            future.set_exception(RuntimeError("fixture worker failed"))
        result=self.planner.poll()
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["reason"],"worker_error")
        self.assertIn("fixture worker failed",result["errors"][0])
        self.assertEqual(result["anchor_state"]["room"],"a")

    def test_fast_exit_does_not_preempt_safe_progress_toward_pending_berry(self):
        self.planner.request(self.state)
        first,second=[row[0] for row in self.executor.submissions]
        exit_route=candidate(room="b")
        exit_route.update(berry_pending=["a:7"],objective_priority=3,goal_progress=1000)
        first.set_result(exit_route)
        self.assertIsNone(self.planner.poll())
        berry_route=candidate(distance=40,mask=16)
        berry_route.update(berry_pending=["a:7"],objective_priority=4,goal_progress=40,
                           active_berry="a:7",goals=[{"kind":"berry","goal":"a:7"}])
        second.set_result(berry_route)
        result=self.planner.poll()
        self.assertEqual(result["active_berry"],"a:7")
        self.assertEqual(result["actions"],[16]*12)

    def test_failed_batch_preserves_pending_goal_and_actual_search_budget(self):
        self.planner.request(self.state,120)
        for future,_,_ in self.executor.submissions:
            future.set_result(dict(can_execute=False,reason="budget",primary_goal="a:7",berry_pending=["a:7"]))
        result=self.planner.poll()
        self.assertEqual(result["primary_goal"],"a:7")
        self.assertEqual(result["berry_pending"],["a:7"])
        self.assertEqual(result["search_seconds"],20)

    def test_close_defaults_to_nonblocking_and_is_idempotent(self):
        self.planner.request(self.state)
        self.planner.close()
        self.planner.close()
        self.assertEqual(self.executor.shutdown_calls,[{"wait":False,"cancel_futures":True}])
        self.assertFalse(self.planner.request(self.state))
        self.assertIsNone(self.planner.poll())
        self.assertFalse(self.planner.busy)
        self.assertTrue(all(f.result_calls==0 for f,_,_ in self.executor.submissions))


if __name__=="__main__":
    unittest.main()
