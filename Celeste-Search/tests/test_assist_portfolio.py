"""Rolling portfolio regressions with controlled futures, never a real pool/game."""
from concurrent.futures import Future
from copy import deepcopy
from unittest.mock import patch
import unittest

from celeste_search.assist import TargetPlanner
from celeste_search.assist_strategies import strategy_spec
from celeste_search.physics import SimState


SPECS=[{"name":name,"family":family,"variant":index,"label":name}
       for index,(name,family) in enumerate((
           ("direct-a","direct"),("direct-b","direct"),
           ("support-a","support_graph"),("support-b","support_graph"),
           ("rested-a","rested_support"),("ordered-a","ordered")))]
ORDER=["direct-a","support-a","rested-a","ordered-a","direct-b","support-b"]


class Clock:
    def __init__(self):
        self.now=100.0

    def __call__(self):
        return self.now


class FakePool:
    def __init__(self,fail_submit=()):
        self.calls=[]
        self.fail_submit=set(fail_submit)
        self.closed=None

    def submit(self,function,config):
        if config.get("strategy") in self.fail_submit:
            raise RuntimeError("submit rejected")
        future=Future()
        future.set_running_or_notify_cancel()
        self.calls.append((deepcopy(config),future))
        return future

    def shutdown(self,**kwargs):
        self.closed=kwargs

    def finish(self,index,*,error=None,result=None):
        future=self.calls[index][1]
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result if result is not None else
                              {"can_execute":False,"reached":False,"reason":"no_route"})


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.pool=FakePool()
        self.planner=TargetPlanner(workers=2,seconds=8,pool=self.pool,clock=self.clock)
        self.addCleanup(self.planner.close)
        self.spec_patch=patch("celeste_search.assist.route_strategy_specs",return_value=deepcopy(SPECS))
        self.spec_patch.start()
        self.addCleanup(self.spec_patch.stop)
        self.state=SimState("a",16,48,on_ground=True)
        self.route={"waypoints":[{"room":"a","x":48,"y":40,"tile_size":8}],
                    "segments":[{"no_dash":False,"no_ground":False}]}

    def request(self,attempt=0):
        return self.planner.request("fixture.bin",self.state,self.route,attempt,[])

    def test_forecast_uses_worker_pool_and_freezes_source_without_starting_search(self):
        source={"room":"a","frame":10,"model_state":{"x":16}}
        future=self.planner.forecast_anchor("fixture.bin",source,7,1800)
        source["model_state"]["x"]=99
        config,submitted=self.pool.calls[0]
        self.assertIs(future,submitted)
        self.assertEqual(config["observation"]["model_state"]["x"],16)
        self.assertEqual((config["request_id"],config["frames"]),(7,1800))
        self.assertEqual(config["generation"],self.planner.epoch)
        self.assertFalse(self.planner.active)
        self.assertEqual(self.planner.stats()["submitted"],0)

    def test_cancel_invalidates_running_forecast_generation_and_discards_completion(self):
        future=self.planner.forecast_anchor("fixture.bin",{"frame":10},7,1800)
        generation=self.pool.calls[0][0]["generation"]
        self.planner.cancel()
        self.assertNotEqual(self.planner.cancel_epoch.value,generation)
        self.pool.finish(0,result={"state":"stale"})
        self.assertIsNone(self.planner.poll())
        self.assertNotIn(future,self.planner.forecasts)

    def ready(self,actions=(1,1)):
        return {"can_execute":True,"reached":True,"actions":list(actions),
                "anchor_state":self.state.to_dict(),
                "expected":[{"room":"a","x":16,"y":48} for _ in range(len(actions)+1)]}

    def test_first_workers_cover_distinct_families_and_inputs_are_frozen(self):
        self.assertTrue(self.request())
        self.assertEqual([config["strategy"] for config,_ in self.pool.calls],ORDER[:2])
        self.assertEqual(self.planner.stats()["strategies"],ORDER)
        self.route["waypoints"][0]["x"]=56
        self.assertEqual(self.pool.calls[0][0]["route"]["waypoints"][0]["x"],48)
        self.assertEqual(self.pool.calls[0][0]["seconds"],8)
        self.assertAlmostEqual(self.pool.calls[1][0]["seconds"],13.6)
        self.assertFalse(self.request())

    def test_region_frontier_gets_fair_slot_and_still_monitors_ordered_routes(self):
        specs=[strategy_spec("ordered_mechanism"),strategy_spec("ordered_mechanism_capability"),
               strategy_spec("ordered_regional"),strategy_spec("direct_greedy")]
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.assertTrue(self.request())
        self.assertEqual([config["strategy"] for config,_ in self.pool.calls],
                         ["ordered_mechanism","ordered_regional"])
        self.assertEqual(specs[2]["family"],"ordered")
        self.assertEqual(self.pool.calls[1][0]["seconds"],90)

    def test_fast_failure_refills_without_waiting_for_unrelated_slow_worker(self):
        self.request()
        slow=self.pool.calls[1][1]
        self.pool.finish(0)
        self.assertIsNone(self.planner.poll())
        self.assertFalse(slow.done())
        self.assertEqual(len(self.pool.calls),3)
        self.assertEqual(self.pool.calls[2][0]["strategy"],"rested-a")
        self.assertEqual(self.planner.stats()["active"],2)
        self.assertEqual(self.planner.stats()["completed"],1)

    def test_deadline_diagnostics_distinguish_untried_running_and_finished(self):
        self.request()
        self.pool.finish(0)
        self.assertIsNone(self.planner.poll())
        self.clock.now=self.planner.deadline
        result=self.planner.poll()
        rows={row["strategy"]:row for row in result["strategy_results"]}
        self.assertEqual(rows["direct-a"]["search_status"],"finished")
        self.assertEqual(rows["support-a"]["search_status"],"running")
        self.assertEqual(rows["support-a"]["reason"],"search_in_progress_at_stop")
        self.assertEqual(rows["direct-b"]["search_status"],"not_started")
        self.assertEqual(rows["direct-b"]["reason"],"strategy_not_started")
        self.assertEqual(set(rows),set(ORDER))

    def test_ordered_strategies_that_all_fit_keep_full_search_slice(self):
        specs=[dict(name=f"ordered-{index}",family="ordered",variant=index) for index in range(2)]
        self.planner.set_deadline(140)
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        self.assertEqual([config["seconds"] for config,_ in self.pool.calls],[39,39])
        self.assertEqual(self.planner.deadline,140)

    def test_queued_ordered_strategies_keep_bounded_slices_for_fairness(self):
        specs=[dict(name=f"ordered-{index}",family="ordered",variant=index) for index in range(3)]
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        self.assertEqual([config["seconds"] for config,_ in self.pool.calls],[8,8])
        self.assertEqual(len(self.planner.pending),1)

    def test_semantic_budget_metadata_keeps_physical_macro_search_within_deadline(self):
        specs=[dict(name=f"ordered-{index}",family="ordered",variant=index,
                    budget_scale=2.5 if index==0 else 1) for index in range(3)]
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        self.assertEqual([config["seconds"] for config,_ in self.pool.calls],[20,8])
        self.assertEqual(self.pool.calls[0][0]["expansions"],int(self.planner.expansions*2.5))

    def test_corridor_time_budget_keeps_node_budget_and_other_workers_unchanged(self):
        specs=[strategy_spec("ordered_corridor"),deepcopy(SPECS[0])]
        self.planner.set_deadline(190)
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        corridor,other=[config for config,_ in self.pool.calls]
        self.assertEqual((corridor["seconds"],corridor["expansions"]),(90,25000))
        self.assertEqual((other["seconds"],other["expansions"]),(8,10000))
        self.assertEqual(self.planner.deadline,190)

    def test_corridor_time_budget_is_clamped_to_actual_request_deadline(self):
        specs=[strategy_spec("ordered_corridor"),deepcopy(SPECS[0])]
        self.planner.set_deadline(110)
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        self.assertEqual(self.pool.calls[0][0]["seconds"],10)
        self.assertEqual(self.planner.deadline,110)

    def test_corridor_time_budget_still_scales_down_with_short_base_budget(self):
        specs=[strategy_spec("ordered_corridor"),deepcopy(SPECS[0])]
        self.planner.seconds=1
        self.planner.set_deadline(145)
        with patch("celeste_search.assist.route_strategy_specs",return_value=specs):
            self.request()
        self.assertEqual(self.pool.calls[0][0]["seconds"],11.25)

    def test_worker_exception_does_not_abort_other_strategies(self):
        self.request()
        self.pool.finish(0,error=RuntimeError("worker failed"))
        self.assertIsNone(self.planner.poll())
        self.assertEqual(len(self.pool.calls),3)
        self.assertEqual(self.planner.results[0]["reason"],"worker_error")
        self.assertTrue(self.planner.active)

    def test_submission_error_continues_filling_capacity(self):
        self.pool.fail_submit.add("direct-a")
        self.request()
        self.assertEqual([config["strategy"] for config,_ in self.pool.calls],["support-a","rested-a"])
        self.assertEqual(self.planner.stats()["completed"],1)
        self.assertEqual(self.planner.stats()["active"],2)

    def test_success_returns_early_and_preserves_winner_stats(self):
        self.request()
        generation=self.planner.cancel_epoch.value
        candidate=self.ready()
        self.pool.finish(0,result=candidate)
        result=self.planner.poll()
        self.assertEqual(result["actions"],[1,1])
        self.assertFalse(self.planner.active)
        self.assertGreater(self.planner.cancel_epoch.value,generation)
        self.assertEqual(self.planner.stats()["winner"],"direct-a")
        self.assertEqual(self.planner.stats()["completed"],1)
        self.assertEqual(self.planner.stats()["total"],6)
        self.assertEqual(self.planner.stats()["queued"],0)
        self.assertEqual(self.planner.stats()["active"],0)
        self.assertEqual(len(self.pool.calls),2)
        self.pool.finish(1,result=self.ready((2,)))
        self.assertIsNone(self.planner.poll())
        self.assertEqual(self.planner.stats()["winner"],"direct-a")

    def test_cancelled_running_jobs_count_against_physical_capacity(self):
        self.request()
        old_generation=self.pool.calls[0][0]["_generation"]
        self.planner.cancel()
        self.assertTrue(self.request(attempt=1))
        self.assertEqual(len(self.pool.calls),2)
        self.assertEqual(self.planner.stats()["active"],0)
        self.assertEqual(self.planner.stats()["queued"],6)
        # A winning stale result must never be admitted into the new request.
        self.pool.finish(0,result=self.ready((1,)))
        self.assertIsNone(self.planner.poll())
        self.assertEqual(len(self.pool.calls),3)
        self.assertGreater(self.pool.calls[2][0]["_generation"],old_generation)
        self.assertEqual(self.planner.stats()["active"],1)
        self.assertEqual(self.planner.stats()["completed"],0)
        self.assertIsNone(self.planner.stats()["winner"])
        self.assertFalse(self.pool.calls[1][1].done())

    def test_cancel_prevents_any_pending_submission(self):
        self.request()
        self.planner.cancel()
        self.pool.finish(0)
        self.pool.finish(1)
        self.assertIsNone(self.planner.poll())
        self.assertEqual(len(self.pool.calls),2)
        self.assertEqual(self.planner.stats()["queued"],0)

    def test_deadline_caps_job_budget_and_stops_pending_work(self):
        self.planner.set_deadline(103)
        self.request()
        self.assertTrue(all(config["seconds"]==3 for config,_ in self.pool.calls))
        self.clock.now=103
        result=self.planner.poll()
        self.assertEqual(result["reason"],"no_route_within_budget")
        self.assertFalse(self.planner.active)
        self.assertEqual(len(self.pool.calls),2)
        self.assertEqual(self.planner.stats()["queued"],0)

    def test_already_expired_deadline_submits_nothing(self):
        self.planner.set_deadline(99)
        self.request()
        self.assertEqual(self.pool.calls,[])
        self.assertEqual(self.planner.poll()["reason"],"no_route_within_budget")

    def test_fail_only_after_entire_portfolio_exhausted(self):
        self.request()
        result=None
        while self.planner.active:
            running=[index for index,(_,future) in enumerate(self.pool.calls) if not future.done()]
            self.assertTrue(running)
            self.pool.finish(running[0])
            result=self.planner.poll()
            if result is not None:
                self.assertEqual(self.planner.stats()["completed"],6)
        self.assertEqual(result["can_execute"],False)
        self.assertEqual([config["strategy"] for config,_ in self.pool.calls],ORDER)
        self.assertEqual(self.planner.stats()["submitted"],6)
        self.assertEqual(self.planner.stats()["active"],0)

    def test_unanimous_unsupported_rejection_is_terminal_not_timeout(self):
        self.request()
        result=None
        while self.planner.active:
            running=[index for index,(_,future) in enumerate(self.pool.calls) if not future.done()]
            self.pool.finish(running[0],result={"can_execute":False,"reason":"unsupported_room_mechanics"})
            result=self.planner.poll()
        self.assertEqual(result["reason"],"unsupported_room_mechanics")
        self.assertTrue(result["terminal_failure"])
        self.assertFalse(result["budget_exhausted"])
        self.assertEqual(len(result["strategy_results"]),len(SPECS))

    def test_one_unsupported_worker_does_not_suppress_a_valid_route(self):
        self.request()
        self.pool.finish(0,result={"can_execute":False,"reason":"unsupported_room_mechanics"})
        self.assertIsNone(self.planner.poll())
        self.pool.finish(1,result=self.ready())
        self.assertTrue(self.planner.poll()["can_execute"])

    def test_non_mapping_response_is_a_worker_failure_then_refilled(self):
        self.request()
        self.pool.calls[0][1].set_result(["invalid"])
        self.assertIsNone(self.planner.poll())
        self.assertEqual(self.planner.results[0]["reason"],"worker_error")
        self.assertEqual(len(self.pool.calls),3)

    def test_partial_candidate_is_not_an_early_winner(self):
        self.request()
        candidate=self.ready()
        candidate["reached"]=False
        self.pool.finish(0,result=candidate)
        self.assertIsNone(self.planner.poll())
        self.assertEqual(len(self.pool.calls),3)
        self.assertIsNone(self.planner.stats()["winner"])

    def test_malformed_executable_candidate_does_not_cancel_valid_concurrent_job(self):
        self.request()
        self.pool.finish(0,result={"can_execute":True,"reached":True,"actions":[]})
        self.assertIsNone(self.planner.poll())
        self.assertTrue(self.planner.active)
        self.pool.finish(1,result=self.ready())
        self.assertEqual(self.planner.poll()["strategy"],"support-a")

    def test_reserved_masks_bad_trace_and_missing_anchor_are_not_winners(self):
        malformed=(
            {"actions":[128]},
            {"actions":[True]},
            {"actions":[-1]},
            {"anchor_state":None},
            {"expected":[]},
            {"can_execute":"yes"},
            {"reached":"yes"},
        )
        for changes in malformed:
            with self.subTest(changes=changes):
                pool=FakePool()
                planner=TargetPlanner(workers=2,seconds=8,pool=pool,clock=self.clock)
                try:
                    planner.request("fixture.bin",self.state,self.route,0,[])
                    candidate=self.ready((1,))
                    candidate.update(changes)
                    pool.finish(0,result=candidate)
                    self.assertIsNone(planner.poll())
                    self.assertTrue(planner.active)
                    self.assertEqual(len(pool.calls),3)
                finally:
                    planner.close()

    def test_result_returned_after_deadline_is_not_an_early_winner(self):
        self.planner.set_deadline(101)
        self.request()
        self.clock.now=102
        self.pool.finish(0,result=self.ready())
        result=self.planner.poll()
        self.assertFalse(result["can_execute"])
        self.assertIsNone(self.planner.stats()["winner"])

    def test_dash_candidate_waits_briefly_and_yields_to_zero_dash_route(self):
        self.request()
        self.pool.finish(0,result=self.ready((32,1)))
        self.assertIsNone(self.planner.poll())
        self.assertTrue(self.planner.active)
        self.clock.now+=.5
        self.pool.finish(1,result=self.ready((1,1,1,1)))
        result=self.planner.poll()
        self.assertEqual(result["strategy"],"support-a")
        self.assertEqual(result["dash_preference"]["dash_count"],0)

    def test_dash_fallback_survives_bounded_comparison_without_waiting_forever(self):
        self.request()
        self.pool.finish(0,result=self.ready((32,1)))
        self.assertIsNone(self.planner.poll())
        self.clock.now+=1.01
        result=self.planner.poll()
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["dash_preference"]["dash_count"],1)

    def test_previously_valid_dash_fallback_is_not_lost_at_deadline(self):
        self.planner.set_deadline(100.5)
        self.request()
        self.pool.finish(0,result=self.ready((32,1)))
        self.assertIsNone(self.planner.poll())
        self.clock.now=100.5
        self.assertTrue(self.planner.poll()["can_execute"])

    def test_cancel_during_preference_window_discards_held_route(self):
        self.request()
        self.pool.finish(0,result=self.ready((32,1)))
        self.assertIsNone(self.planner.poll())
        self.planner.cancel()
        self.clock.now+=2
        self.assertIsNone(self.planner.poll())
        self.assertIsNone(self.planner.preference_candidate)

    def test_exact_handoff_can_flush_existing_or_just_completed_dash_result(self):
        for already_polled in (False,True):
            with self.subTest(already_polled=already_polled):
                self.request()
                index=len(self.pool.calls)-2
                self.pool.finish(index,result=self.ready((32,1)))
                if already_polled:
                    self.assertIsNone(self.planner.poll())
                self.planner.finish_preference_window()
                result=self.planner.poll()
                self.assertTrue(result["can_execute"])
                for number,(_,future) in enumerate(self.pool.calls):
                    if not future.done():
                        self.pool.finish(number)


if __name__=="__main__":
    unittest.main()
