"""Human-assist controller tests: fake workers, offline telemetry, no game."""
from dataclasses import replace
from concurrent.futures import Future
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from celeste_search.assist import AssistController, AssistOptions, PASSIVE, TargetPlanner
from celeste_search.assist_planning import target_reached
from celeste_search.assist_validation import make_assist_anchor
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DT, RIGHT, SimState, Simulator
from celeste_search.room_planning import action_fingerprint
from celeste_search.routes import snapshot


def observation(state, frame=None, **changes):
    result = state.to_dict()
    result.update(model_state=state.to_dict(), model_state_unavailable=[],
                  on_safe_ground=state.on_ground, input_phase="before_virtual_inputs",
                  control_mode="manual", human_input_mask=0, human_menu_input=False,
                  human_input_epoch=3, input_bindings_supported=True,
                  delta_time=DT, raw_delta_time=DT, spawn_id=1, deaths=0,
                  area_id=1, area_mode=0,
                  room_entities={"entities":[], "do_not_load":[], "unavailable":[]})
    if frame is not None:
        result["frame"] = frame
    result.update(changes)
    return result


class FakeTargetPlanner:
    def __init__(self, simulator):
        self.simulator = simulator
        self.epoch = 0
        self.pending = False
        self.requests = []
        self.ready = []
        self.cancellations = 0

    def request(self, map_path, state, target, attempt, forbidden):
        if self.pending:
            return False
        self.epoch += 1
        self.pending = True
        self.requests.append(dict(map_path=map_path, state=state, target=dict(target),
                                  attempt=attempt, forbidden=set(forbidden), epoch=self.epoch))
        return True

    def cancel(self):
        self.epoch += 1
        self.pending = False
        self.cancellations += 1

    def poll(self):
        while self.ready:
            epoch, result = self.ready.pop(0)
            if epoch == self.epoch:
                self.pending = False
                return result
        return None

    def publish(self, result, epoch=None):
        self.ready.append((self.epoch if epoch is None else epoch, result))

    def candidate(self, masks=None):
        request = self.requests[-1]
        state = request["state"]
        states, actions, boundaries = [state], [], []
        for mask in ([RIGHT]*160 if masks is None else masks):
            next_state = self.simulator.step(state, mask)
            actions.append(mask)
            states.append(next_state)
            if next_state.room != state.room:
                boundaries.append(len(actions))
            state = next_state
            if target_reached(state, request["target"]):
                break
        return dict(can_execute=True, reached=True, actions=actions,
                    expected=[snapshot(item) for item in states],
                    final_state=state.to_dict(), anchor_state=states[0].to_dict(),
                    fingerprint=action_fingerprint(actions), target=request["target"],
                    allow_room_continuation=bool(boundaries), boundary_indices=boundaries), states


class ControlledPlannerPool:
    """Real portfolio bookkeeping with controlled Futures and no processes."""
    def __init__(self):
        self.calls=[]

    def submit(self,function,config):
        future=Future()
        future.set_running_or_notify_cancel()
        self.calls.append((config,future))
        return future

    def shutdown(self,**kwargs):
        pass


class AssistControllerTests(unittest.TestCase):
    def setUp(self):
        rows = ("0"*16,)*8 + ("1"*16,)*2
        room = Room("a", 0, 0, 128, 80, rows, (), ((24.,64.),))
        second = replace(room, name="b", x=128, spawns=((152.,64.),))
        self.map = MapData("assist-controller", {"a":room, "b":second}, Path("fixture.bin"))
        self.sim = Simulator(self.map)
        self.start = self.sim.initial_state("a")
        self.planner = FakeTargetPlanner(self.sim)
        self.controller = AssistController(self.map, self.planner)
        self.target = {"room":"a", "x":88, "y":56, "tile_size":8}

    def prime(self, target=None, state=None):
        state = state or self.start
        self.controller.confirm(target or self.target)
        for frame in range(1,4):
            self.assertEqual(self.controller.observe(observation(state, frame)), PASSIVE)
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(self.controller.mode, "planning")

    def activate(self, target=None, state=None, masks=None):
        state = state or self.start
        self.prime(target, state)
        candidate, states = self.planner.candidate(masks)
        self.planner.publish(candidate)
        action = self.controller.observe(observation(state, 4))
        self.assertEqual(action, candidate["actions"][0])
        self.assertEqual(self.controller.mode, "executing")
        return candidate, states

    def test_default_is_always_physical_passthrough_without_confirm(self):
        for frame in range(1,30):
            action = self.controller.observe(observation(self.start, frame, human_input_mask=RIGHT))
            self.assertEqual(action, 128)
        self.assertEqual(self.controller.mode, "manual")
        self.assertFalse(self.planner.requests)
        self.assertIsNone(self.controller.target)

    def test_confirm_only_starts_background_search_without_taking_control(self):
        state = replace(self.start, x=40, berries=frozenset({"previous:5"}))
        self.prime(state=state)
        for frame in range(4,100):
            self.assertEqual(self.controller.observe(observation(state, frame)), PASSIVE)
        request = self.planner.requests[0]
        self.assertEqual(request["state"].x, 40)
        self.assertEqual(request["state"].berries, state.berries)
        self.assertEqual(request["target"], self.target)
        self.assertEqual(len(self.planner.requests), 1)

    def test_actual_anchor_must_still_match_before_first_action(self):
        self.prime()
        candidate, _ = self.planner.candidate()
        self.planner.publish(candidate)
        moved = replace(self.start, rem_x=.01)
        self.assertEqual(self.controller.observe(observation(moved, 4)), PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertNotEqual(self.controller.mode, "executing")
        self.assertTrue(any(item["event"] == "anchor_invalidated" for item in self.controller.events))

    def test_unsupported_mechanics_reports_reason_without_repeating_search(self):
        self.prime()
        self.planner.publish(dict(can_execute=False,reason="unsupported_room_mechanics",
                                  terminal_failure=True,budget_exhausted=False))
        self.assertEqual(self.controller.observe(observation(self.start,4)),PASSIVE)
        self.assertEqual(self.controller.mode,"failed")
        self.assertIn("尚未支持的机关",self.controller.message)
        self.assertNotIn("时间预算",self.controller.message)
        for frame in range(5,15):
            self.assertEqual(self.controller.observe(observation(self.start,frame)),PASSIVE)
        self.assertEqual(len(self.planner.requests),1)

    def test_replaced_request_cannot_accept_previous_goal_result(self):
        self.prime()
        candidate, _ = self.planner.candidate()
        old_epoch = self.planner.epoch
        self.controller.confirm({"room":"b", "x":144, "y":56})
        self.planner.publish(candidate, old_epoch)
        for frame in range(4,8):
            self.assertEqual(self.controller.observe(observation(self.start, frame)), PASSIVE)
        self.assertEqual(self.controller.target["room"], "b")
        self.assertIsNone(self.controller.plan)

    def test_manual_epoch_change_invalidates_search_even_at_same_pose(self):
        self.prime()
        candidate, _ = self.planner.candidate()
        old_epoch = self.planner.epoch
        self.planner.publish(candidate, old_epoch)
        action = self.controller.observe(observation(self.start, 4, human_input_epoch=4))
        self.assertEqual(action, PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertTrue(any(item.get("reason") == "runtime_epoch_changed" for item in self.controller.events))

    def test_manual_input_during_execution_immediately_returns_control(self):
        _, states = self.activate()
        action = self.controller.observe(observation(states[1], 5,
                                         human_input_mask=RIGHT, human_input_epoch=4))
        self.assertEqual(action, PASSIVE)
        self.assertEqual(self.controller.mode, "manual")
        self.assertEqual(self.controller.result, "cancelled")
        for frame in range(6,10):
            self.assertEqual(self.controller.observe(observation(states[1], frame)), PASSIVE)
        self.assertEqual(len(self.planner.requests), 1)

    def test_same_pose_new_human_epoch_cancels_execution(self):
        _, states = self.activate()
        self.assertEqual(self.controller.observe(observation(states[1], 5, human_input_epoch=4)), PASSIVE)
        self.assertEqual(self.controller.mode, "manual")

    def test_success_requires_actual_selected_tile_and_returns_passthrough(self):
        candidate, states = self.activate()
        self.assertNotEqual(self.controller.result, "reached")
        self.assertTrue(candidate["reached"])  # The worker's claim alone is insufficient.
        for index, state in enumerate(states[1:], start=1):
            action = self.controller.observe(observation(state, index+4, control_mode="assist"))
            if target_reached(state, self.target):
                self.assertEqual(action, PASSIVE)
                break
            self.assertLess(action, 128)
            self.assertNotEqual(self.controller.mode, "success")
        self.assertEqual(self.controller.mode, "success")
        self.assertEqual(self.controller.result, "reached")
        self.assertEqual(self.controller.observe(observation(states[-1], 999)), PASSIVE)

    def test_false_worker_success_does_not_award_target_after_actions_end(self):
        _, states = self.activate(masks=[RIGHT])
        self.assertFalse(target_reached(states[-1], self.target))
        self.assertEqual(self.controller.observe(observation(states[-1], 5, control_mode="assist")), PASSIVE)
        self.assertEqual(self.controller.mode, "waiting")
        self.assertTrue(self.controller.recovery["allow_completion"])
        self.assertNotEqual(self.controller.result, "reached")

    def test_search_failures_stop_at_bounded_attempts(self):
        self.controller.options = AssistOptions(max_attempts=2)
        self.prime()
        for frame in (4,8):
            self.planner.publish({"can_execute":False, "reason":"no_route_within_budget"})
            self.assertEqual(self.controller.observe(observation(self.start, frame)), PASSIVE)
            if frame==4:
                for idle_frame in range(5,8):
                    self.controller.observe(observation(self.start,idle_frame))
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.controller.mode, "failed")
        self.assertEqual(self.controller.observe(observation(self.start, 6)), PASSIVE)

    def test_elapsed_budget_returns_human_control_without_an_action(self):
        with patch("celeste_search.assist.time.monotonic", return_value=10):
            self.controller.confirm(self.target)
        expired=self.controller.started+self.controller.options.max_search_seconds+1
        with patch("celeste_search.assist.time.monotonic", return_value=expired):
            self.assertEqual(self.controller.observe(observation(self.start, 1)), PASSIVE)
        self.assertEqual(self.controller.mode, "failed")
        self.assertFalse(self.planner.requests)

    def test_explicit_cancel_discards_ready_result_and_stays_manual(self):
        self.prime()
        candidate, _ = self.planner.candidate()
        self.planner.publish(candidate)
        self.controller.cancel()
        for frame in range(4,10):
            self.assertEqual(self.controller.observe(observation(self.start, frame)), PASSIVE)
        self.assertEqual(self.controller.mode, "manual")
        self.assertEqual(self.controller.result, "cancelled")
        self.assertIsNone(self.controller.plan)

    def test_planned_crossroom_input_tail_continues_without_replanning(self):
        target = {"room":"b", "x":144, "y":56, "tile_size":8}
        candidate, states = self.activate(target=target)
        self.assertTrue(candidate["boundary_indices"])
        boundary = candidate["boundary_indices"][0]
        for index, state in enumerate(states[1:], start=1):
            action = self.controller.observe(observation(state, index+4, control_mode="assist"))
            if target_reached(state, target):
                self.assertEqual(action, PASSIVE)
                break
            self.assertLess(action, 128)
            self.assertEqual(self.controller.mode, "executing")
            if index == boundary:
                self.assertEqual(state.room, "b")
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(self.controller.mode, "success")

    def test_natural_respawn_replans_only_from_current_room_actual_position(self):
        self.activate()
        respawn = replace(self.start, room="b", x=168)
        action = self.controller.observe(observation(respawn, 5, spawn_id=2, deaths=1))
        self.assertEqual(action, PASSIVE)
        self.assertEqual(self.controller.mode, "waiting")
        self.assertEqual(self.controller.execution_retries, 1)
        for frame in range(6,10):
            self.assertEqual(self.controller.observe(observation(respawn, frame, spawn_id=2, deaths=1)), PASSIVE)
        self.assertEqual(self.planner.requests[-1]["state"].room, "b")
        self.assertEqual(self.planner.requests[-1]["state"].x, 168)

    def test_dead_player_releases_inputs_before_spawn_counter_changes(self):
        _, states = self.activate()
        for frame in range(5,8):
            answer = self.controller.observe(observation(states[1], frame, dead=True, control_mode="assist"))
            self.assertEqual(answer, PASSIVE)
            self.assertEqual(self.controller.index, 0)
        self.assertEqual(self.controller.observe(observation(self.start, 8, spawn_id=2, deaths=1)), PASSIVE)
        self.assertIsNone(self.controller.plan)

    def test_cutscene_returns_human_control_without_waiting_for_route_end(self):
        _, states = self.activate()
        self.assertEqual(self.controller.observe(observation(states[1], 5, in_cutscene=True)), PASSIVE)
        self.assertEqual(self.controller.mode, "manual")
        self.assertIsNone(self.controller.plan)

    def test_already_at_selected_tile_succeeds_without_worker_or_movement(self):
        self.controller.confirm({"room":"a", "x":24, "y":56})
        self.assertEqual(self.controller.observe(observation(self.start, 1)), PASSIVE)
        self.assertEqual(self.controller.mode, "success")
        self.assertFalse(self.planner.requests)

    def test_invalid_target_never_starts_worker(self):
        self.controller.confirm({"room":"a", "x":24, "y":64})
        for frame in range(1,6):
            self.assertEqual(self.controller.observe(observation(self.start, frame)), PASSIVE)
        self.assertEqual(self.controller.mode, "failed")
        self.assertFalse(self.planner.requests)

    def test_missing_candidate_origin_cannot_issue_first_input(self):
        self.prime()
        candidate, _ = self.planner.candidate()
        del candidate["anchor_state"]
        self.planner.publish(candidate)
        self.assertEqual(self.controller.observe(observation(self.start, 4)), PASSIVE)
        self.assertIsNone(self.controller.plan)

    def test_worker_cannot_issue_control_protocol_bytes_or_incomplete_trace(self):
        for changes in ({"actions":[254]}, {"actions":[255]}, {"actions":[128]},
                        {"actions":[True]}, {"expected":[]}):
            with self.subTest(changes=changes):
                self.planner = FakeTargetPlanner(self.sim)
                self.controller = AssistController(self.map, self.planner)
                self.prime()
                candidate, _ = self.planner.candidate(masks=[RIGHT])
                candidate.update(changes)
                self.planner.publish(candidate)
                self.assertEqual(self.controller.observe(observation(self.start, 4)), PASSIVE)
                self.assertIsNone(self.controller.plan)

    def test_repeated_divergences_warn_and_continue_without_reset_commands(self):
        self.controller.options = AssistOptions(max_execution_retries=0)
        _, states = self.activate()
        moved = replace(states[1], x=states[1].x+30)
        commands = [self.controller.observe(observation(moved, 5, control_mode="assist"))]
        commands += [self.controller.observe(observation(moved, frame)) for frame in range(6,10)]
        self.assertEqual(self.controller.mode, "executing")
        self.assertEqual(commands, [RIGHT]*5)
        self.assertEqual(len(self.controller.execution_warnings),1)
        self.assertFalse(self.controller.forbidden)
        self.assertNotIn(254, commands)
        self.assertNotIn(255, commands)

    def prime_future(self):
        """A short explicit future anchor exercises admission, without sleeps."""
        with patch.object(self.controller,"prepare_anchor",side_effect=lambda obs:
                make_assist_anchor(self.sim,obs,request_id=self.controller.request_id,future_frames=6)):
            self.prime()
        self.assertEqual(self.controller.anchor["target_frame"],9)
        candidate,_=self.planner.candidate()
        self.planner.publish(candidate)
        return candidate

    def test_future_candidate_waits_passively_until_exact_predicted_frame(self):
        candidate=self.prime_future()
        for frame in range(4,9):
            self.assertEqual(self.controller.observe(observation(self.start,frame)),PASSIVE)
            self.assertIsNone(self.controller.plan)
            self.assertIsNotNone(self.controller.pending_candidate)
        self.assertEqual(self.controller.observe(observation(self.start,9)),candidate["actions"][0])
        self.assertEqual(self.controller.mode,"executing")
        self.assertIsNone(self.controller.pending_candidate)
        self.assertEqual(len(self.planner.requests),1)

    def prime_real_portfolio_future(self):
        pool=ControlledPlannerPool()
        # Keep wall time frozen inside the one-second quality window; only
        # exact-frame admission may release this otherwise held dash result.
        planner=TargetPlanner(workers=2,pool=pool,clock=lambda:100.)
        self.addCleanup(planner.close)
        controller=AssistController(self.map,planner)
        controller.confirm(self.target)
        with patch.object(controller,"prepare_anchor",side_effect=lambda obs:
                make_assist_anchor(self.sim,obs,request_id=controller.request_id,future_frames=6)):
            for frame in range(1,4):
                self.assertEqual(controller.observe(observation(self.start,frame)),PASSIVE)
        self.assertEqual(controller.anchor["target_frame"],9)
        self.assertEqual(len(pool.calls),2)
        candidate_maker=FakeTargetPlanner(self.sim)
        candidate_maker.request(self.map.source,SimState.from_dict(controller.anchor["state"]),
                                self.target,0,set())
        candidate,states=candidate_maker.candidate(masks=[RIGHT|DASH]+[RIGHT]*80)
        self.assertTrue(target_reached(states[-1],self.target))
        return controller,planner,pool,candidate,states

    def test_future_handoff_flushes_real_dash_preference_at_exact_frame_only(self):
        for already_held in (True,False):
            with self.subTest(already_held=already_held):
                controller,planner,pool,candidate,states=self.prime_real_portfolio_future()
                with patch.object(planner,"finish_preference_window",wraps=planner.finish_preference_window) as finish:
                    if already_held:
                        pool.calls[0][1].set_result(candidate)
                    for frame in range(4,9):
                        self.assertEqual(controller.observe(observation(self.start,frame)),PASSIVE)
                        self.assertIsNone(controller.plan)
                        self.assertIsNone(controller.pending_candidate)
                    finish.assert_not_called()
                    if already_held:
                        self.assertIsNotNone(planner.preference_candidate)
                        self.assertEqual(planner.preference_until,101.)
                    else:
                        # Also cover a result first drained by poll on the
                        # admission frame, before any comparison window exists.
                        pool.calls[0][1].set_result(candidate)
                    self.assertEqual(controller.observe(observation(self.start,9)),RIGHT|DASH)
                    finish.assert_called_once_with()
                    self.assertEqual(controller.mode,"executing")
                    self.assertEqual(controller.observe(observation(states[1],control_mode="assist")),candidate["actions"][1])
                    self.assertEqual(controller.mode,"executing")
                    self.assertFalse(any(event.get("reason")=="future_frame_missed"
                                         for event in controller.events))

    def test_cancelled_future_cannot_flush_held_or_late_dash_preference(self):
        controller,planner,pool,candidate,_=self.prime_real_portfolio_future()
        pool.calls[0][1].set_result(candidate)
        self.assertEqual(controller.observe(observation(self.start,4)),PASSIVE)
        self.assertIsNotNone(planner.preference_candidate)
        controller.cancel()
        self.assertIsNone(planner.preference_candidate)
        # A second worker can complete after cancellation; its generation may
        # never become executable when the old future frame arrives.
        pool.calls[1][1].set_result(candidate)
        with patch.object(planner,"finish_preference_window",wraps=planner.finish_preference_window) as finish:
            for frame in (9,10):
                self.assertEqual(controller.observe(observation(self.start,frame)),PASSIVE)
            finish.assert_not_called()
        self.assertEqual(controller.mode,"manual")
        self.assertIsNone(controller.plan)
        planner.finish_preference_window()
        self.assertIsNone(planner.poll())

    def prime_subpixel_future(self):
        """Use the real drift gate and forecast admission, with no clock room."""
        self.controller.options=AssistOptions(future_anchor_frames=40)
        states=[replace(self.start,frame=0,vy=0.0002117157,rem_y=-0.0313462839)]
        for _ in range(45):
            states.append(self.sim.step(states[-1],0))
        submissions=[]
        def forecast(map_path,obs,request_id,frames):
            submissions.append((obs["frame"],frames))
            future=Future()
            future.set_result(make_assist_anchor(self.sim,obs,
                request_id=request_id,future_frames=frames))
            return future
        self.planner.forecast_anchor=forecast
        self.controller.confirm(self.target)
        for state in states[1:5]:
            self.assertEqual(self.controller.observe(observation(state)),PASSIVE)
        self.assertEqual(submissions,[(3,40)])
        self.assertEqual(len(self.planner.requests),1)
        self.assertEqual(self.controller.anchor["mode"],"future")
        self.assertEqual(self.controller.anchor["target_frame"],43)
        candidate,_=self.planner.candidate()
        self.planner.publish(candidate)
        return candidate,states,submissions

    def test_real_subpixel_drift_gate_waits_past_29_frames_until_exact_future_takeover(self):
        candidate,states,submissions=self.prime_subpixel_future()
        for state in states[5:43]:
            self.assertEqual(self.controller.observe(observation(state)),PASSIVE)
            self.assertIsNone(self.controller.plan)
        self.assertGreater(abs(states[42].rem_y-states[3].rem_y),0.0001)
        self.assertEqual(self.controller.observe(observation(states[43])),candidate["actions"][0])
        self.assertEqual(self.controller.mode,"executing")
        self.assertEqual(submissions,[(3,40)])
        self.assertFalse(any(event["event"]=="anchor_invalidated" for event in self.controller.events))

    def test_subpixel_future_human_input_or_cancel_cannot_emit_candidate_action(self):
        for cancel in (False,True):
            with self.subTest(cancel=cancel):
                self.setUp()
                _,states,_=self.prime_subpixel_future()
                self.assertEqual(self.controller.observe(observation(states[5])),PASSIVE)
                self.assertIsNotNone(self.controller.pending_candidate)
                if cancel:
                    self.controller.cancel()
                    obs=observation(states[6])
                else:
                    obs=observation(states[6],human_input_mask=RIGHT,human_input_epoch=4)
                self.assertEqual(self.controller.observe(obs),PASSIVE)
                self.assertIsNone(self.controller.plan)
                self.assertIsNone(self.controller.pending_candidate)
                self.assertEqual(self.controller.observe(observation(states[43],human_input_epoch=4)),PASSIVE)
                self.assertNotEqual(self.controller.mode,"executing")

    def test_subpixel_future_rejects_actual_remainder_error_and_missed_frame(self):
        for late in (False,True):
            with self.subTest(late=late):
                self.setUp()
                _,states,_=self.prime_subpixel_future()
                for state in states[5:43]:
                    self.assertEqual(self.controller.observe(observation(state)),PASSIVE)
                actual=states[44] if late else replace(states[43],rem_y=states[43].rem_y+0.001)
                self.assertEqual(self.controller.observe(observation(actual)),PASSIVE)
                self.assertIsNone(self.controller.plan)
                self.assertIsNone(self.controller.pending_candidate)
                self.assertNotEqual(self.controller.mode,"executing")

    def test_pending_subpixel_forecast_is_not_replaced_by_newly_stationary_idle_state(self):
        self.controller.confirm(self.target)
        drift=replace(self.start,vy=0.0002117157,rem_y=-0.0313462839)
        future=Future()
        with patch.object(self.planner,"forecast_anchor",create=True,return_value=future) as submit:
            self.assertIsNone(self.controller.prepare_anchor(observation(drift,3)))
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,4)))
            self.assertIs(self.controller.anchor_forecast,future)
            self.assertEqual(submit.call_count,1)
            self.controller.cancel()
            self.assertTrue(future.cancelled())

    def test_human_epoch_invalidates_a_ready_future_candidate_before_takeover(self):
        self.prime_future()
        self.controller.observe(observation(self.start,4))
        self.assertEqual(self.controller.observe(observation(self.start,5,human_input_epoch=4)),PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertIsNone(self.controller.pending_candidate)
        self.assertNotEqual(self.controller.mode,"executing")

    def test_future_frame_missed_discards_candidate_instead_of_replaying_late(self):
        self.prime_future()
        self.controller.observe(observation(self.start,4))
        self.assertEqual(self.controller.observe(observation(self.start,10)),PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertIsNone(self.controller.pending_candidate)
        self.assertTrue(any(event.get("reason")=="future_frame_missed"
                            for event in self.controller.events))

    def test_future_candidate_does_not_survive_cancel_or_replacement(self):
        self.prime_future()
        self.controller.observe(observation(self.start,4))
        self.controller.cancel()
        self.assertIsNone(self.controller.pending_candidate)
        self.assertEqual(self.controller.observe(observation(self.start,9)),PASSIVE)
        self.controller.pending_candidate={"stale":True}
        self.controller.confirm(self.target)
        self.assertIsNone(self.controller.pending_candidate)

    def test_failed_future_search_reports_unsupported_without_waiting_for_clock(self):
        self.prime_future()
        self.planner.ready.clear()
        self.planner.publish(dict(can_execute=False,reason="unsupported_room_mechanics",
                                  terminal_failure=True))
        self.assertEqual(self.controller.observe(observation(self.start,4)),PASSIVE)
        self.assertEqual(self.controller.mode,"failed")
        self.assertIsNone(self.controller.pending_candidate)

    def test_clock_forecast_runs_outside_the_frame_response_path(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassetteBlock",1,64,48,16,8),))
        self.controller.map=replace(self.map,rooms={"a":room})
        started,release=Event(),Event()
        def blocked_forecast(*args,**kwargs):
            started.set()
            if not release.wait(2):
                raise RuntimeError("test did not release forecast")
            return {"request_id":self.controller.request_id}
        with patch("celeste_search.assist.make_assist_anchor",side_effect=blocked_forecast):
            try:
                self.assertIsNone(self.controller.prepare_anchor(observation(self.start,3)))
                self.assertTrue(started.wait(1))
                future=self.controller.anchor_forecast
                self.assertFalse(future.done())
                # A second tick also returns immediately, without another job.
                self.assertIsNone(self.controller.prepare_anchor(observation(self.start,4)))
                self.assertIs(self.controller.anchor_forecast,future)
                self.controller.cancel()
                self.assertIsNone(self.controller.anchor_forecast)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=1),{"request_id":self.controller.request_id})
            self.assertIsNone(self.controller.anchor_forecast)

    def test_process_forecast_is_not_waited_on_and_cancelled_result_cannot_replace_new_one(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassetteBlock",1,64,48,16,8),))
        self.controller.map=replace(self.map,rooms={"a":room})
        first,second=Future(),Future()
        first.set_running_or_notify_cancel()
        with patch.object(self.planner,"forecast_anchor",create=True,
                          side_effect=[first,second]) as submit, \
             patch("celeste_search.assist.make_assist_anchor",side_effect=AssertionError("local forecast")):
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,3)))
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,4)))
            self.assertEqual(submit.call_count,1)
            self.controller.cancel()
            self.controller.confirm(self.target)
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,5)))
            first.set_result({"state":"old request"})
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,6)))
            self.assertIs(self.controller.anchor_forecast,second)
            self.assertEqual(submit.call_count,2)
            self.controller.cancel()
            self.assertTrue(second.cancelled())

    def test_late_retry_forecast_fits_remaining_request_budget(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassetteBlock",1,64,48,16,8),))
        self.controller.map=replace(self.map,rooms={"a":room})
        self.controller.started=100
        deadline=self.controller.started+self.controller.options.max_search_seconds
        now=deadline-20
        with patch.object(self.planner,"forecast_anchor",create=True,return_value=Future()) as submit, \
             patch("celeste_search.assist.time.monotonic",return_value=now):
            self.assertIsNone(self.controller.prepare_anchor(observation(self.start,3)))
        self.assertEqual(submit.call_args.args[-1],1170)
        self.assertLess(now+submit.call_args.args[-1]/60,deadline)
        self.controller.clear_forecast()

    def test_pickup_waypoint_reports_unsupported_before_any_search_or_takeover(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassette",1,80,40,16,16),))
        self.controller.map=replace(self.map,rooms={"a":room})
        route={"waypoints":[{"room":"a","x":80,"y":40,"tile_size":8},self.target],
               "segments":[{"no_dash":False,"no_ground":False}]*2}
        self.controller.confirm_route(route)
        self.assertEqual(self.controller.mode,"failed")
        self.assertIn("第 1 个目标",self.controller.message)
        self.assertIn("手动拾取",self.controller.message)
        self.assertEqual(self.controller.observe(observation(self.start)),PASSIVE)
        self.assertEqual(self.planner.requests,[])

    def test_adjacent_partial_pickup_tile_remains_searchable(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassette",1,80,40,16,16),))
        self.controller.map=replace(self.map,rooms={"a":room})
        self.controller.confirm_route({"waypoints":[{"room":"a","x":88,"y":40,"tile_size":8}],
                                       "segments":[{"no_dash":False,"no_ground":False}]})
        self.assertEqual(self.controller.mode,"waiting")

    def test_verified_stopped_blocks_use_idle_anchor_without_future_countdown(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassetteBlock",1,64,48,16,8),))
        self.controller.map=replace(self.map,rooms={"a":room})
        with patch("celeste_search.assist.state_from_observation",
                   return_value=SimpleNamespace(room="a",cassette_stopped=True)), \
             patch("celeste_search.assist.make_assist_anchor",return_value={"mode":"idle"}) as make:
            self.assertEqual(self.controller.prepare_anchor(observation(self.start)),{"mode":"idle"})
        self.assertNotIn("future_frames",make.call_args.kwargs)
        self.assertIsNone(self.controller.anchor_forecast)

    def test_verified_stopped_room_allows_former_pickup_target(self):
        room=replace(self.map.rooms["a"],entities=(Entity("cassette",1,80,40,16,16),))
        self.controller.map=replace(self.map,rooms={"a":room})
        self.controller.observation=observation(self.start)
        with patch("celeste_search.assist.state_from_observation",
                   return_value=SimpleNamespace(room="a",cassette_stopped=True)):
            self.controller.confirm_route({"waypoints":[{"room":"a","x":80,"y":40,"tile_size":8}],
                                           "segments":[{"no_dash":False,"no_ground":False}]})
        self.assertEqual(self.controller.mode,"waiting")

    def test_cassette_phase_divergence_warns_but_actual_target_can_succeed(self):
        candidate,states=self.activate()
        marker=dict(cassette_clock={"beat_index":1,"actual_depth":-0.000001},cassette_blocks=[],
                    player_depth=0,player_actual_depth=0.0)
        candidate["expected"][1].update(marker)
        with patch("celeste_search.assist.cassette_snapshot_from_observation",
                   return_value={**marker,"cassette_clock":{**marker["cassette_clock"],"beat_index":2}}):
            action=self.controller.observe(observation(states[-1],5,control_mode="assist"))
        self.assertEqual(action,PASSIVE)
        self.assertEqual(self.controller.mode,"success")
        self.assertEqual(self.controller.result,"reached")
        self.assertTrue(self.controller.execution_warnings)

    def test_forecast_worker_error_fails_request_without_breaking_passive_service(self):
        self.controller.confirm(self.target)
        with patch.object(self.controller,"prepare_anchor",side_effect=RuntimeError("forecast failure")):
            for frame in range(1,4):
                self.assertEqual(self.controller.observe(observation(self.start,frame)),PASSIVE)
        self.assertEqual(self.controller.mode,"failed")
        self.assertTrue(any(event["event"]=="anchor_forecast_failed" for event in self.controller.events))
        self.assertEqual(self.controller.observe(observation(self.start,4)),PASSIVE)


if __name__ == "__main__":
    unittest.main()
