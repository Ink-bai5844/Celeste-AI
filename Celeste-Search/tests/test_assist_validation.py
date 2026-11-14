"""Target and asynchronous handoff tests use only the independent simulator."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist_validation import (
    AssistValidationError, admit_assist_candidate, make_assist_anchor,
    observe_assist_anchor, target_from_world, validate_tile_target,
)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DT, Simulator
from celeste_search.anchor import can_plan_while_idle


def fixture(entities=(), *, x=-128, y=-80, height=80):
    rows = ("0"*16,)*8 + ("1"*16,)*2
    room = Room("a", x, y, 128, height, rows, tuple(entities), ((x+16., y+64.),))
    data = MapData("assist-fixture", {"a": room}, Path("assist-fixture.bin"))
    return data, Simulator(data)


def observed(state, **changes):
    result = state.to_dict()
    result.update(model_state=state.to_dict(), model_state_unavailable=[],
                  on_safe_ground=state.on_ground, input_phase="before_virtual_inputs",
                  control_mode="manual", human_input_mask=0, human_menu_input=False,
                  input_bindings_supported=True, delta_time=DT, raw_delta_time=DT,
                  human_input_epoch=4, spawn_id=2, deaths=1, area_id=1, area_mode=0,
                  room_entities={"entities": [], "do_not_load": [], "unavailable": []})
    result.update(changes)
    return result


class TargetValidationTests(unittest.TestCase):
    def test_negative_world_click_and_airborne_tile_are_retained(self):
        data, _ = fixture()
        target = target_from_world(data, "a", -113.2, -77.5)
        self.assertEqual(target, {"room": "a", "x": -120, "y": -80, "tile_size": 8})
        self.assertEqual(validate_tile_target(data, target), target)

    def test_solid_and_room_boundary_rejected_without_snapping(self):
        data, _ = fixture()
        for target, reason in (({"room":"a", "x":-120, "y":-16}, "target_is_solid"),
                               ({"room":"a", "x":0, "y":-40}, "target_outside_room"),
                               ({"room":"a", "x":-120, "y":-88}, "target_outside_room"),
                               ({"room":"missing", "x":0, "y":0}, "unknown_target_room")):
            with self.subTest(target=target), self.assertRaises(AssistValidationError) as context:
                validate_tile_target(data, target)
            self.assertEqual(context.exception.reason, reason)

    def test_partial_native_184_last_tile_allowed_but_outside_click_rejected(self):
        data, _ = fixture(x=0, y=0, height=184)
        self.assertEqual(target_from_world(data, "a", 10, 179)["y"], 176)
        with self.assertRaises(AssistValidationError):
            target_from_world(data, "a", 10, 180)

    def test_breakable_or_dangerous_tiles_remain_searchable(self):
        for kind in ("dashBlock", "spikesUp", "jumpThru"):
            data, _ = fixture([Entity(kind, 1, -120, -80, 8, 8)])
            self.assertEqual(validate_tile_target(data, {"room":"a", "x":-120, "y":-80})["x"], -120)

    def test_permanent_entity_only_rejects_fully_covered_tile(self):
        for width in (8, 4):
            data, _ = fixture([Entity("solid", 1, -120, -80, width, 8)])
            if width == 8:
                with self.assertRaises(AssistValidationError):
                    validate_tile_target(data, {"room":"a", "x":-120, "y":-80})
            else:
                validate_tile_target(data, {"room":"a", "x":-120, "y":-80})

    def test_nonfinite_boolean_unaligned_and_wrong_size_are_invalid(self):
        data, _ = fixture()
        for changes in ({"x":float("nan")}, {"x":float("inf")}, {"x":True},
                        {"x":-121}, {"tile_size":16}, {"tile_size":True}):
            with self.subTest(changes=changes), self.assertRaises(AssistValidationError):
                validate_tile_target(data, {"room":"a", "x":-120, "y":-80, **changes})


class AssistAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.data, self.sim = fixture()
        self.state = replace(self.sim.initial_state("a"), frame=100)
        self.obs = observed(self.state)

    def anchor(self, **kwargs):
        return make_assist_anchor(self.sim, self.obs, request_id="job1", **kwargs)

    def admit(self, anchor, obs=None, **kwargs):
        return admit_assist_candidate(self.data, anchor, self.obs if obs is None else obs,
                                      request_id="job1", **kwargs)

    def test_idle_survives_time_but_uses_fresh_actual_frame(self):
        anchor = self.anchor()
        later = observed(replace(self.state, frame=900))
        result = self.admit(anchor, later, candidate_state=anchor["state"])
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["state"]["frame"], 900)
        self.assertEqual(result["state"]["x"], self.state.x)

    def test_tiny_native_velocity_requires_timed_anchor_before_remainder_accumulates(self):
        # Native grounded 9b STATE after a diagonal dash: less than one pixel
        # of movement, but rem_y crosses the existing 1e-4 gate after 29 ticks.
        for vy in (0.0002117157, 0.00010585785):
            with self.subTest(vy=vy):
                state = replace(self.state, vy=vy, rem_y=-0.0313462839)
                obs = observed(state)
                self.assertTrue(can_plan_while_idle(obs))
                self.assertEqual(self.sim.step(state, 0).y, state.y)
                with self.assertRaises(AssistValidationError) as context:
                    make_assist_anchor(self.sim, obs, request_id="job1")
                self.assertEqual(context.exception.reason, "timed_projection_required")

    def test_stationary_decimal_telemetry_remainder_does_not_require_forecast(self):
        # JSON's shortened decimal and its native float32 value differ in
        # Python float64, without any physical movement.
        state = replace(self.state, rem_x=-0.124969825, rem_y=-0.0313462839)
        self.assertNotEqual(self.sim.step(state, 0).rem_y, state.rem_y)
        anchor = make_assist_anchor(self.sim, observed(state), request_id="job1")
        self.assertEqual(anchor["mode"], "idle")

    def test_tiny_remainder_drift_uses_exact_future_state_without_relaxed_admission(self):
        state = replace(self.state, vy=0.0002117157, rem_y=-0.0313462839)
        obs = observed(state)
        anchor = make_assist_anchor(self.sim, obs, request_id="job1", future_frames=1800)
        expected = state
        for _ in range(1800):
            expected = self.sim.step(expected, 0)
        self.assertEqual((expected.x, expected.y), (state.x, state.y))
        self.assertGreater(abs(expected.rem_y-state.rem_y), 0.0001)
        before = self.admit(deepcopy(anchor), observed(replace(expected, frame=expected.frame-1)))
        self.assertEqual(before["reason"], "future_frame_not_reached")
        admitted = self.admit(deepcopy(anchor), observed(expected), candidate_state=anchor["state"])
        self.assertTrue(admitted["accepted"], admitted)
        mismatch = self.admit(deepcopy(anchor), observed(replace(expected, rem_y=expected.rem_y+0.001)))
        self.assertFalse(mismatch["accepted"])
        self.assertEqual(mismatch["different_fields"], ["rem_y"])

    def test_future_projection_accepts_long_budget_but_bounds_it_to_one_minute(self):
        self.assertEqual(self.anchor(future_frames=3600)["target_frame"], 3700)
        for frames in (-1, 3601, True, 1.5):
            with self.subTest(frames=frames), self.assertRaises(AssistValidationError) as context:
                self.anchor(future_frames=frames)
            self.assertEqual(context.exception.reason, "invalid_projection_frames")

    def test_existing_bridge_without_inactive_hop_fields_can_plan_and_admit(self):
        obs = deepcopy(self.obs)
        for field in ("climb_hop_solid", "climb_hop_x", "climb_hop_y"):
            obs.pop(field, None)
            obs["model_state"].pop(field, None)
        anchor = make_assist_anchor(self.sim, obs, request_id="job1")
        self.assertEqual(anchor["state"]["climb_hop_solid"], "")
        later = deepcopy(obs)
        later["frame"] += 20
        later["model_state"]["frame"] += 20
        result = self.admit(anchor, later, candidate_state=anchor["state"])
        self.assertTrue(result["accepted"], result)

    def test_existing_bridge_cannot_plan_during_unknown_active_hop(self):
        obs = deepcopy(self.obs)
        obs["model_state"].update(force_frames=8, force_x=0)
        for field in ("climb_hop_solid", "climb_hop_x", "climb_hop_y"):
            obs.pop(field, None)
            obs["model_state"].pop(field, None)
        with self.assertRaises(AssistValidationError) as context:
            make_assist_anchor(self.sim, obs, request_id="job1")
        self.assertEqual(context.exception.reason, "invalid_runtime_state")
        self.assertIn("climb_hop_attachment_unavailable", str(context.exception))

    def test_no_spawn_replacement_or_two_pixel_tolerance(self):
        for changes in ({"x":self.state.x+1}, {"rem_x":.001}, {"dashes":0}, {"facing":-1}):
            with self.subTest(changes=changes):
                result = self.admit(self.anchor(), observed(replace(self.state, **changes)))
                self.assertFalse(result["accepted"], result)

    def test_human_press_permanently_invalidates_even_after_return(self):
        anchor = self.anchor()
        status = observe_assist_anchor(self.data, anchor,
                  observed(self.state, human_input_mask=1, human_input_epoch=5), request_id="job1")
        self.assertFalse(status["valid"])
        self.assertFalse(self.admit(anchor)["accepted"])

    def test_physical_epoch_catches_unobserved_press_then_release(self):
        result = self.admit(self.anchor(), observed(self.state, human_input_epoch=5))
        self.assertEqual(result["reason"], "runtime_epoch_changed")

    def test_menu_press_blocks_handoff_and_missing_input_fails_closed(self):
        for changes in ({"human_menu_input":True}, {"human_menu_input":None},
                        {"human_input_mask":None}, {"control_mode":"assist"}):
            with self.subTest(changes=changes), self.assertRaises(AssistValidationError):
                make_assist_anchor(self.sim, observed(self.state, **changes), request_id=1)

    def test_old_control_event_does_not_block_new_manual_request(self):
        obs = observed(self.state, control_event="human_input")
        anchor = make_assist_anchor(self.sim, obs, request_id=1)
        self.assertTrue(admit_assist_candidate(self.data, anchor, obs, request_id=1)["accepted"])

    def test_unsupported_bindings_or_changed_game_clock_cannot_take_over(self):
        for changes in ({"input_bindings_supported":False}, {"delta_time":DT/2},
                        {"raw_delta_time":DT*2}, {"delta_time":1/60}, {"delta_time":None}):
            with self.subTest(changes=changes), self.assertRaises(AssistValidationError):
                make_assist_anchor(self.sim, observed(self.state, **changes), request_id=1)

    def test_respawn_death_session_request_and_frame_rewind_reject(self):
        for changes in ({"spawn_id":3}, {"deaths":2}, {"area_id":2}, {"frame":99}, {"dead":True}):
            with self.subTest(changes=changes):
                self.assertFalse(self.admit(self.anchor(), observed(self.state, **changes))["accepted"])
        result = admit_assist_candidate(self.data, self.anchor(), self.obs, request_id="job2")
        self.assertEqual(result["reason"], "request_changed")

    def test_unknown_private_field_cannot_turn_into_default(self):
        obs = deepcopy(self.obs)
        del obs["model_state"]["rem_y"]
        with self.assertRaises(AssistValidationError):
            make_assist_anchor(self.sim, obs, request_id=1)

    def test_current_environment_change_rejects_even_with_unchanged_player(self):
        anchor = self.anchor()
        obs = deepcopy(self.obs)
        obs["room_entities"]["entities"] = [{"id":"a:3", "kind":"refill", "respawn_timer":1.0}]
        result = self.admit(anchor, obs)
        self.assertEqual(result["reason"], "environment_changed")

    def test_wrong_worker_start_rejected_before_execution(self):
        wrong = replace(self.state, x=self.state.x+8)
        result = self.admit(self.anchor(), candidate_state=wrong)
        self.assertEqual(result["reason"], "candidate_origin_changed")

    def test_future_airborne_admission_requires_exact_frame_and_state(self):
        air = replace(self.state, y=self.state.y-25, vy=0, on_ground=False, grace=0)
        source = observed(air)
        anchor = make_assist_anchor(self.sim, source, request_id="job1", future_frames=2)
        self.assertFalse(anchor["state"]["on_ground"])
        before = self.admit(anchor, observed(self.sim.step(air, 0)))
        self.assertEqual(before, {"accepted":False, "reason":"future_frame_not_reached", "waiting":True})
        expected = self.sim.step(self.sim.step(air, 0), 0)
        result = self.admit(anchor, observed(expected))
        self.assertTrue(result["accepted"], result)

    def test_future_late_result_and_model_mismatch_are_rejected(self):
        for late in (False, True):
            anchor = self.anchor(future_frames=2)
            state = self.sim.step(self.sim.step(self.state, 0), 0)
            state = replace(state, frame=state.frame+1) if late else replace(state, rem_y=.01)
            result = self.admit(anchor, observed(state))
            self.assertEqual(result["reason"], "future_frame_missed" if late else "projected_state_mismatch")

    def test_make_does_not_mutate_or_advance_source_observation(self):
        before = deepcopy(self.obs)
        self.anchor(future_frames=10)
        self.assertEqual(self.obs, before)


if __name__ == "__main__":
    unittest.main()
