"""Failure evidence survives cancellation without polling or copying full plans."""
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch
import json
import unittest

from celeste_search.assist import (AssistController, AssistOptions, PASSIVE, TargetPlanner,
                                   _bounded_strategy_results)
from celeste_search.maps import MapData, Room
from celeste_search.physics import Simulator
from test_assist_controller import FakeTargetPlanner, ControlledPlannerPool, observation


class FailureDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        room=Room("a",0,0,128,80,("0"*16,)*8+("1"*16,)*2,(),((24.,64.),))
        self.data=MapData("failure-diagnostics",{"a":room},Path("fixture.bin"))
        self.sim=Simulator(self.data)
        self.state=self.sim.initial_state("a")
        self.target=dict(room="a",x=88,y=56,tile_size=8)

    def prime(self,controller):
        controller.confirm(self.target)
        for frame in range(1,4):
            self.assertEqual(controller.observe(observation(self.state,frame)),PASSIVE)
        self.assertEqual(controller.mode,"planning")

    def test_failed_result_survives_cancel_and_is_frozen_once_for_publication(self):
        planner=FakeTargetPlanner(self.sim)
        controller=AssistController(self.data,planner,AssistOptions(max_attempts=1))
        self.prime(controller)
        rows=[dict(strategy="ordered_corridor",reason="no_corridor_route_within_budget",
                   expansions=2418,wall_seconds=35.0,
                   strategy_details=dict(backtracks=8,corridor_attempts=[
                       dict(portal_index=2,reason="physical_leg_budget",expanded=2418)]))]
        planner.publish(dict(can_execute=False,budget_exhausted=True,strategy_results=rows))
        self.assertEqual(controller.observe(observation(self.state,4)),PASSIVE)
        self.assertEqual(controller.mode,"failed")
        recorded=controller.status()["strategy_results"]
        self.assertEqual(recorded,rows)
        rows[0]["strategy_details"]["backtracks"]=999
        self.assertEqual(recorded[0]["strategy_details"]["backtracks"],8)
        for frame in range(5,8):
            self.assertEqual(controller.observe(observation(self.state,frame)),PASSIVE)
            self.assertIs(controller.status()["strategy_results"],recorded)
        failure=next(row for row in controller.events if row["event"]=="offline_attempt_failed")
        self.assertIs(failure["strategy_results"],recorded)

    def test_controller_deadline_collects_finished_current_jobs_without_polling(self):
        pool=ControlledPlannerPool()
        planner=TargetPlanner(workers=2,pool=pool,clock=lambda:100.)
        self.addCleanup(planner.close)
        controller=AssistController(self.data,planner,AssistOptions(max_attempts=1))
        self.prime(controller)
        completed,pending=[future for _,future in pool.calls]
        completed.set_result(dict(can_execute=False,reason="no_route",expansions=123,
                                  actions=[1]*20000,expected=[{}]*20001))
        stale=Future();stale.set_result(dict(strategy="old",reason="old_epoch"))
        planner.jobs[stale]=dict(epoch=planner.epoch-1,strategy="old")
        planner.results.append(dict(strategy="already_drained",reason="budget",expansions=456))
        with patch.object(planner,"poll",side_effect=AssertionError("must not poll at deadline")), \
             patch.object(pending,"result",side_effect=AssertionError("must not wait on pending job")), \
             patch("celeste_search.assist.time.monotonic",
                   return_value=controller.started+controller.options.max_search_seconds+1):
            self.assertEqual(controller.observe(observation(self.state,4)),PASSIVE)
        rows=controller.status()["strategy_results"]
        self.assertEqual(controller.mode,"failed")
        finished=[row for row in rows if row["search_status"]=="finished"]
        running=[row for row in rows if row["search_status"]=="running"]
        self.assertEqual({row["reason"] for row in finished},{"budget","no_route"})
        self.assertEqual({row["expansions"] for row in finished},{123,456})
        self.assertEqual(len(running),1)
        self.assertEqual(running[0]["reason"],"search_in_progress_at_stop")
        self.assertNotIn("expansions",running[0])
        self.assertEqual(len(rows),3)
        self.assertEqual(len(pool.calls),2)
        self.assertEqual(planner.results,[])
        self.assertTrue(all("actions" not in row and "expected" not in row for row in rows))

    def test_new_or_invalid_request_clears_prior_request_diagnostics(self):
        controller=AssistController(self.data,FakeTargetPlanner(self.sim))
        controller.strategy_results=[dict(strategy="old",reason="budget")]
        controller.confirm(self.target)
        self.assertEqual(controller.status()["strategy_results"],[])
        controller.strategy_results=[dict(strategy="old",reason="budget")]
        controller.confirm_route(dict(waypoints=[],segments=[]))
        self.assertEqual(controller.status()["strategy_results"],[])

    def test_large_recursive_worker_details_are_bounded_and_omit_trajectories(self):
        cycle={"reason":"nested"};cycle["cycle"]=cycle
        raw=[dict(strategy="策略"*500,reason="失败"*500,error="错误"*500,
                  wall_seconds=float("nan"),expansions=10**1000,
                  expected=[object()]*5000,actions=[1]*5000,
                  strategy_details=dict(cycle=cycle,actions=[object()]*5000,
                      attempts=[dict(reason="预算"*500,expanded=i) for i in range(1000)]))
             for _ in range(100)]
        result=_bounded_strategy_results(raw)
        encoded=json.dumps(result,ensure_ascii=False,allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(result),48)
        self.assertLessEqual(len(encoded),65536)
        self.assertTrue(all(len(json.dumps(row,ensure_ascii=False).encode("utf-8"))<=4096
                            for row in result))
        self.assertNotIn(b'"actions"',encoded)
        self.assertNotIn(b'"expected"',encoded)
        self.assertIsNone(result[0]["wall_seconds"])
        self.assertIsNone(result[0]["expansions"])
        malformed=_bounded_strategy_results([dict(strategy="valid",reason={"huge":raw},
                                                 expansions={"huge":raw})])
        self.assertEqual(malformed,[dict(strategy="valid")])


if __name__=="__main__":
    unittest.main()
