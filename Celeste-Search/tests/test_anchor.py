"""Stale asynchronous candidate rejection, with no real game execution."""
import unittest

from celeste_search.anchor import anchor_matches, can_plan_while_idle, candidate_compatible
from celeste_search.physics import SimState


def idle(**changes):
    state = SimState("1", 19, 144, on_ground=True, grace=6)
    obs = state.to_dict()
    obs.update(on_safe_ground=True, input_phase="before_virtual_inputs", model_state=state.to_dict())
    obs.update(changes)
    return obs


class AnchorTests(unittest.TestCase):
    def test_idle_floor_accepts_and_frame_elapsed_does_not_stale(self):
        anchor = idle(frame=7)
        observed = idle(frame=600)
        anchor["runtime_timers_seconds"] = {"varJumpTimer": -0.2}
        observed["runtime_timers_seconds"] = {"varJumpTimer": -4.5}
        self.assertTrue(can_plan_while_idle(anchor))
        self.assertTrue(anchor_matches(observed, anchor))

    def test_air_climb_dash_unsafe_ground_and_motion_refused(self):
        for changes in ({"on_ground": False}, {"state": 1}, {"state": 2},
                        {"on_safe_ground": False}, {"vx": 2}, {"vy": -3},
                        {"dead": True}, {"transitioning": True}, {"stamina": 70}):
            self.assertFalse(can_plan_while_idle(idle(**changes)), changes)

    def test_positive_timers_and_autojump_wait_to_settle(self):
        for field in ("dash_cooldown", "refill_cooldown", "jump_buffer", "dash_buffer",
                      "force_frames", "lift_timer", "var_jump"):
            obs = idle()
            obs["model_state"][field] = 1
            self.assertFalse(can_plan_while_idle(obs), field)
        obs = idle()
        obs["model_state"]["auto_jump_active"] = True
        self.assertFalse(can_plan_while_idle(obs))
        obs["model_state"]["auto_jump_active"] = False
        self.assertTrue(can_plan_while_idle(obs))

    def test_pending_berry_waits_for_actual_confirmation(self):
        self.assertFalse(can_plan_while_idle(idle(following=["1:7"])))
        self.assertTrue(can_plan_while_idle(idle(berries=["1:7"])))
        self.assertFalse(anchor_matches(idle(berries=["1:7"]), idle()))

    def test_neutral_direction_history_normalized_but_button_edges_not(self):
        anchor = idle()
        anchor["model_state"]["prev_input"] = 1
        self.assertTrue(anchor_matches(idle(), anchor))
        for mask in (16, 32, 64):
            anchor["model_state"]["prev_input"] = mask
            self.assertFalse(can_plan_while_idle(anchor))

    def test_ground_grace_refresh_is_not_a_stale_state(self):
        anchor, observed = idle(), idle()
        anchor["model_state"]["grace"] = 0
        observed["model_state"]["grace"] = 6
        self.assertTrue(anchor_matches(observed, anchor))

    def test_leftover_climb_timer_does_not_prevent_normal_idle(self):
        anchor, observed = idle(), idle()
        anchor["model_state"]["climb_wait"] = 5
        observed["model_state"]["climb_wait"] = 5
        self.assertTrue(can_plan_while_idle(observed))
        self.assertTrue(anchor_matches(observed, anchor))

    def test_position_inventory_and_subpixel_changes_reject(self):
        anchor = idle()
        self.assertTrue(anchor_matches(idle(x=20), anchor))
        self.assertFalse(anchor_matches(idle(x=22), anchor))
        self.assertFalse(anchor_matches(idle(dashes=0), anchor))
        observed = idle()
        observed["model_state"]["rem_x"] = 0.25
        self.assertFalse(anchor_matches(observed, anchor))

    def test_epoch_changes_reject_even_if_player_matches(self):
        for name in ("spawn_id", "deaths", "room_epoch", "environment_fingerprint"):
            self.assertFalse(anchor_matches(idle(**{name: 2}), idle(**{name: 1})))

    def test_known_moving_or_ticking_environment_is_not_idle(self):
        obs = idle()
        obs["model_state"]["mechanisms"] = [["1:9", 2, 12, 20, 70, 0]]
        self.assertFalse(can_plan_while_idle(obs))
        obs["model_state"]["mechanisms"] = []
        obs["model_state"]["refill_timers"] = [["1:10", 20]]
        self.assertFalse(can_plan_while_idle(obs))

    def test_unknown_environment_does_not_invent_motion_on_static_floor(self):
        obs = idle()
        obs["model_state_unavailable"] = ["mechanisms", "broken", "refill_timers", "fled"]
        for name in obs["model_state_unavailable"]:
            obs["model_state"].pop(name)
        self.assertTrue(can_plan_while_idle(obs))

    def test_candidate_initial_private_state_is_checked(self):
        state = SimState("1", 19, 144, on_ground=True, grace=6)
        self.assertTrue(candidate_compatible(idle(), state))
        wrong = state.to_dict()
        wrong["facing"] = -1
        self.assertFalse(candidate_compatible(idle(), wrong))
        wrong = state.to_dict()
        wrong["dash_buffer"] = 2
        self.assertFalse(candidate_compatible(idle(), wrong))

    def test_candidate_without_epochs_still_requires_separate_raw_anchor_match(self):
        state = SimState("1", 19, 144, on_ground=True, grace=6)
        current = idle(spawn_id=4, deaths=1)
        self.assertTrue(candidate_compatible(current, state))
        original = idle(spawn_id=3, deaths=0)
        self.assertFalse(anchor_matches(current, original))
        self.assertFalse(candidate_compatible(current, original))

    def test_missing_or_invalid_metadata_fails_closed(self):
        obs = idle()
        del obs["model_state"]["rem_x"]
        self.assertFalse(can_plan_while_idle(obs))
        self.assertFalse(can_plan_while_idle(idle(x=float("nan"))))
        self.assertFalse(anchor_matches(idle(), idle(), tolerance=-1))
        obs = idle()
        obs["model_state"]["dash_cooldown"] = 0.1
        self.assertFalse(can_plan_while_idle(obs))


if __name__ == "__main__":
    unittest.main()
