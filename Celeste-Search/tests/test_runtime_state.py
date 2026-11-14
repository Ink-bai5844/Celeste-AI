"""Telemetry mapping tests; no game process or sockets are involved."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState
from celeste_search.runtime_state import RuntimeStateError, state_from_observation, state_readiness


def fixture_map():
    a = Room("a", 0, 0, 320, 184, (), (), ((24.0, 168.0),), {})
    b = replace(a, name="b", x=320, spawns=((340.0, 168.0),))
    return MapData("fixture", {"a": a, "b": b}, Path("fixture.bin"))


def observation(**changes):
    data = {"room": "a", "x": 103.0, "y": 120.0, "vx": 90.0, "vy": 0.0,
            "dashes": 1, "stamina": 86.5, "on_ground": True, "state": 0,
            "berries": [], "following": [], "frame": 713, "dead": False,
            "input_phase": "before_virtual_inputs"}
    data.update(changes)
    return data


def air_metadata(**changes):
    data = {"rem_x": 0.25, "rem_y": -0.1, "prev_input": 17, "grace": 0,
            "var_jump": 7, "var_speed": -105.0, "auto_jump": 0, "auto_jump_active": False, "dash_cooldown": 3,
            "refill_cooldown": 2, "facing": 1, "climb_wait": 0,
            "force_x": 0, "force_frames": 0}
    data.update(changes)
    return data


class RuntimeStateTests(unittest.TestCase):
    def test_actual_position_and_global_frame_never_replaced_by_spawn(self):
        state = state_from_observation(fixture_map(), observation(room="b", x=321.0, y=67.0))
        self.assertEqual((state.room, state.x, state.y, state.frame), ("b", 321.0, 67.0, 713))
        self.assertNotEqual(state.x, fixture_map().rooms["b"].spawns[0][0])

    def test_runtime_core_overrides_model_duplicates_and_predicted_collection(self):
        previous = SimState("a", 10, 20, berries=frozenset({"a:old"}), frame=99)
        obs = observation(berries=["a:actual"], following=["a:actual", "a:pending"],
                          model_state={"x": -999, "frame": 0, "berries": ["a:fake"]})
        state = state_from_observation(fixture_map(), obs, previous)
        self.assertEqual(state.x, 103)
        self.assertEqual(state.frame, 713)
        self.assertEqual(state.berries, frozenset({"a:actual"}))
        self.assertEqual(state.following, frozenset({"a:pending"}))

    def test_same_room_preserves_known_environment_only_until_observed_override(self):
        previous = SimState("a", 10, 20, mechanisms=(("a:1", 2, 9, 40.0, 80.0, 0.0),),
                            broken=frozenset({"a:2"}), refill_timers=(("a:3", 60),),
                            fled=frozenset({"a:4"}))
        state = state_from_observation(fixture_map(), observation(), previous)
        self.assertEqual(state.mechanisms, previous.mechanisms)
        self.assertEqual(state.broken, previous.broken)
        self.assertEqual(state.refill_timers, previous.refill_timers)
        live = state_from_observation(fixture_map(), observation(model_state={"broken": []}), previous)
        self.assertFalse(live.broken)

    def test_room_change_clears_previous_room_environment(self):
        previous = SimState("a", 10, 20, mechanisms=(("a:1", 2, 9, 40.0, 80.0, 0.0),),
                            broken=frozenset({"a:2"}), refill_timers=(("a:3", 60),),
                            fled=frozenset({"a:4"}), lift_timer=5, prev_input=17)
        state = state_from_observation(fixture_map(), observation(room="b", x=324), previous)
        self.assertFalse(state.mechanisms or state.broken or state.refill_timers or state.fled)
        self.assertEqual(state.lift_timer, 0)

    def test_airborne_private_timers_keep_frame_units_and_subpixels(self):
        obs = observation(on_ground=False, vy=-105.0, model_state=air_metadata())
        self.assertTrue(state_readiness(obs)["ready"])
        state = state_from_observation(fixture_map(), obs)
        self.assertEqual(state.var_jump, 7)
        self.assertEqual(state.dash_cooldown, 3)
        self.assertEqual(state.rem_x, 0.25)
        self.assertEqual(state.prev_input, 17)

    def test_missing_airborne_timers_and_dash_coroutines_are_refused(self):
        for obs in (observation(on_ground=False), observation(state=2, model_state=air_metadata(dash_frames=5))):
            self.assertFalse(state_readiness(obs)["ready"])
            with self.assertRaises(RuntimeStateError):
                state_from_observation(fixture_map(), obs)

    def test_unavailable_essential_field_cannot_use_placeholder(self):
        obs = observation(on_ground=False, model_state=air_metadata(), model_state_unavailable=["var_jump"])
        ready = state_readiness(obs)
        self.assertFalse(ready["ready"])
        self.assertIn("var_jump", ready["missing_fields"])

    def test_follower_order_and_negative_collection_stagger_survive(self):
        obs = observation(following=["a:2", "a:1"],
                          model_state={"follow_order": ["a:2", "a:1"], "collect_delay": 0,
                                       "collect_ground": -9},
                          model_state_approximate=["collect_ground"])
        state = state_from_observation(fixture_map(), obs)
        self.assertEqual(state.follow_order, ("a:2", "a:1"))
        self.assertEqual(state.collect_ground, -9)
        self.assertTrue(state_readiness(obs)["warnings"])

    def test_dead_cutscene_invalid_phase_and_unknown_room_refused(self):
        for changes in ({"dead": True}, {"state": 11}, {"transitioning": True},
                        {"input_phase": "after_update"}, {"x": float("nan")},
                        {"frame": 2.5}, {"room": "missing"}):
            with self.assertRaises(RuntimeStateError):
                state_from_observation(fixture_map(), observation(**changes))

    def test_floating_second_timers_are_not_silently_treated_as_frames(self):
        obs = observation(model_state={"grace": 0.1})
        self.assertEqual(state_readiness(obs)["reason"], "invalid_observation")
        with self.assertRaises(RuntimeStateError):
            state_from_observation(fixture_map(), obs)

    def test_indefinite_autojump_flag_and_timer_are_independent(self):
        obs = observation(on_ground=False, vy=-30,
                          model_state=air_metadata(auto_jump_active=True, auto_jump=0, var_jump=0))
        state = state_from_observation(fixture_map(), obs)
        self.assertTrue(state.auto_jump_active)
        self.assertEqual(state.auto_jump, 0)
        self.assertEqual(state.var_jump, 0)
        transient = observation(model_state={"auto_jump_active": False, "auto_jump": 4})
        state = state_from_observation(fixture_map(), transient)
        self.assertFalse(state.auto_jump_active)
        self.assertEqual(state.auto_jump, 4)

    def test_legacy_airborne_timer_without_boolean_is_refused(self):
        model = air_metadata()
        del model["auto_jump_active"]
        ready = state_readiness(observation(on_ground=False, model_state=model))
        self.assertFalse(ready["ready"])
        self.assertIn("auto_jump_active", ready["missing_fields"])

    def test_collision_continuation_fields_are_preserved(self):
        obs = observation(model_state={"dash_attack": 4, "wall_retention": 3,
                          "wall_retained": 170.0, "hop_wait_x": -1, "hop_wait_x_speed": -100.0})
        state = state_from_observation(fixture_map(), obs)
        self.assertEqual((state.dash_attack, state.wall_retention, state.wall_retained), (4, 3, 170))
        self.assertEqual((state.hop_wait_x, state.hop_wait_x_speed), (-1, -100))

    def test_native_catch_during_flight_overrides_predicted_fled_identity(self):
        previous = SimState("a", 10, 20, fled=frozenset({"a:3", "a:4"}))
        state = state_from_observation(fixture_map(),
                                      observation(following=["a:3"], berries=["a:4"]), previous)
        self.assertFalse(state.fled)
        self.assertEqual(state.following, frozenset({"a:3"}))

    def test_observed_environment_overrides_predicted_breaks_refills_and_flee(self):
        original = fixture_map()
        a = replace(original.rooms["a"], entities=(
            Entity("dashBlock", 1, 40, 80, 16, 16),
            Entity("refill", 2, 60, 80, 0, 0),
            Entity("strawberry", 3, 80, 80, 0, 0, {"winged": True})))
        mapping = replace(original, rooms={"a": a, "b": original.rooms["b"]})
        previous = SimState("a", 10, 20, refill_timers=(("a:2", 140),))
        snapshot = {"entities": [
            {"id": "a:1", "room": "a", "present": False},
            {"id": "a:2", "room": "a", "present": True, "respawn_timer": 0.5},
            {"id": "a:3", "room": "a", "present": True, "flying_away": True}],
            "do_not_load": []}
        state = state_from_observation(mapping, observation(room_entities=snapshot), previous)
        self.assertEqual(state.broken, frozenset({"a:1"}))
        self.assertEqual(state.refill_timers, (("a:2", 30),))
        self.assertEqual(state.fled, frozenset({"a:3"}))
        self.assertFalse(state.berries)
        snapshot["entities"][0]["present"] = True
        snapshot["entities"][1]["respawn_timer"] = 0
        snapshot["entities"][2]["flying_away"] = False
        respawned = state_from_observation(mapping, observation(room_entities=snapshot), state)
        self.assertFalse(respawned.broken or respawned.refill_timers or respawned.fled)

    def test_mover_position_is_measured_but_phase_is_not_claimed_exact(self):
        original = fixture_map()
        a = replace(original.rooms["a"], entities=(
            Entity("zipMover", 1, 40, 80, 32, 16, {}, ((140.0, 80.0),)),))
        mapping = replace(original, rooms={"a": a})
        snapshot = {"entities": [{"id": "a:1", "room": "a", "present": True,
                    "x": 90.0, "y": 80.0, "lift_vx": 100.0, "lift_vy": 0.0}], "do_not_load": []}
        obs = observation(room_entities=snapshot)
        state = state_from_observation(mapping, obs)
        self.assertEqual(state.mechanisms[0][3:5], (90.0, 80.0))
        ready = state_readiness(obs)
        self.assertFalse(ready["exact"])
        self.assertTrue(any("coroutine timing is still approximate" in warning for warning in ready["warnings"]))

    def test_unobserved_predicted_wall_break_is_not_native_evidence(self):
        original = fixture_map()
        a = replace(original.rooms["a"], entities=(Entity("dashBlock", 1, 40, 80, 16, 16),))
        mapping = replace(original, rooms={"a": a})
        previous = SimState("a", 10, 20, broken=frozenset({"a:1"}))
        state = state_from_observation(mapping, observation(room_entities={"entities": []}), previous)
        self.assertFalse(state.broken)
        observed = state_from_observation(mapping, observation(
            room_entities={"entities": [], "do_not_load": ["a:1"]}), previous)
        self.assertEqual(observed.broken, frozenset({"a:1"}))

    def test_malformed_runtime_entity_snapshot_cannot_anchor(self):
        obs = observation(room_entities={"entities": [{"id": "a:1", "lift_vy": "unknown"}]})
        self.assertFalse(state_readiness(obs)["ready"])

    def test_native_zip_wait_and_progress_are_taken_from_verified_iterator(self):
        original = fixture_map()
        a = replace(original.rooms["a"], entities=(
            Entity("zipMover", 1, 40, 80, 32, 16, {}, ((140.0, 80.0),)),))
        mapping = replace(original, rooms={"a": a})
        iterator = {"iterator_type": "Celeste.ZipMover+<Sequence>d__24",
                    "fields": {"<>1__state": 4, "<at>5__3": 1.0}}
        coroutine = {"active": True, "wait_timer": 0.1, "stack": [iterator]}
        record = {"id": "a:1", "room": "a", "present": True,
                  "x": 140.0, "y": 80.0, "coroutines": [coroutine]}
        snapshot = {"entities": [record]}
        state = state_from_observation(mapping, observation(room_entities=snapshot))
        self.assertEqual(state.mechanisms[0][1:3], (3, 7))
        iterator["fields"] = {"<>1__state": 5, "<at>5__3": 0.25}
        state = state_from_observation(mapping, observation(room_entities=snapshot))
        self.assertEqual(state.mechanisms[0][1:3], (4, 30))


if __name__ == "__main__":
    unittest.main()
