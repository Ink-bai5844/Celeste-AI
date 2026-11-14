"""Offline invariants for the planning model; these are not game conformance tests."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP, SimState, Simulator


def flat_map(entities=(), *, width=40, height=23):
    rows = tuple("0" * width for _ in range(height - 2)) + ("1" * width,) * 2
    room = Room("test", 0, 0, width * 8, height * 8, rows, tuple(entities), ((24.0, (height - 2) * 8.0),), {})
    return MapData("fixture", {room.name: room}, Path("fixture.bin"))


class PhysicsTests(unittest.TestCase):
    def test_standing_and_run_speed(self):
        sim = Simulator(flat_map())
        start = sim.initial_state()
        self.assertTrue(start.on_ground)
        self.assertEqual(sim.roll(start, 0, 60).y, start.y)
        running = sim.roll(start, RIGHT, 60)
        self.assertEqual(running.vx, 90.0)
        self.assertEqual(running.y, start.y)
        self.assertGreater(running.x, start.x + 80)

    def test_hold_jump_higher_than_tap(self):
        sim = Simulator(flat_map())
        start = sim.initial_state()
        held = sim.trace(start, [JUMP] * 35)
        tapped = sim.trace(start, [JUMP] + [0] * 34)
        self.assertLess(min(s.y for s in held), min(s.y for s in tapped) - 8)
        self.assertEqual(start.y, sim.roll(held[-1], 0, 60).y)

    def test_dash_is_edge_triggered_and_consumes_stock(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(), y=110, on_ground=False, grace=0)
        first = sim.step(start, RIGHT | DASH)
        self.assertEqual(first.dashes, 0)
        self.assertEqual(first.state, 2)
        end = sim.roll(first, RIGHT | DASH, 8)
        self.assertGreater(end.x, start.x + 20)
        self.assertEqual(end.dashes, 0)
        self.assertEqual(end.frame, 9)

    def test_dash_begin_preserves_decaying_variable_jump_timer(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(), y=110, vy=-100, on_ground=False, grace=0,
                        var_jump=9, var_speed=-105)
        first = sim.step(start, RIGHT | DOWN | DASH)
        self.assertEqual((first.state, first.var_jump), (2, 8))
        self.assertEqual(sim.roll(first, RIGHT | DOWN, 3).var_jump, 5)

    def test_late_diagonal_dash_landing_boosts_after_attack_has_expired(self):
        platform = Entity("jumpThru", 1, 48, 128, 32, 0)
        sim = Simulator(flat_map([platform]))
        start = replace(sim.initial_state(), x=43, y=128, vx=272.999878, vy=160,
                        on_ground=False, grace=0, dashes=0, state=0,
                        dash_frames=0, dash_attack=0, dash_dx=.707106769, dash_dy=.707106769,
                        auto_jump_active=True, prev_input=RIGHT)
        landed = sim.step(start, RIGHT | JUMP)
        self.assertEqual((landed.y, landed.vy), (128, 0))
        self.assertEqual((landed.dash_dx, landed.dash_dy), (1, 0))
        self.assertTrue(landed.ducking)
        self.assertAlmostEqual(landed.vx, 322.399841, places=4)
        jumped = sim.step(landed, RIGHT | JUMP)
        self.assertAlmostEqual(jumped.vx, 355.733154, places=4)
        self.assertEqual(jumped.vy, -105)
        horizontal = sim.step(replace(start, dash_dy=0), RIGHT)
        self.assertFalse(horizontal.ducking)
        self.assertAlmostEqual(horizontal.vx, 268.666534, places=4)

    def test_spike_death_and_no_tunneling_through_floor(self):
        spike = Entity("spikesUp", 2, 32, 168, 16, 0)
        sim = Simulator(flat_map([spike]))
        falling = replace(sim.initial_state(), x=40, y=145, vy=160, on_ground=False)
        result = sim.roll(falling, 0, 20)
        self.assertTrue(result.dead)
        self.assertEqual(result.reason, "hazard")

    def test_strawberry_needs_follow_delay_and_safe_ground(self):
        berry = Entity("strawberry", 7, 24, 159, 0, 0)
        sim = Simulator(flat_map([berry]))
        first = sim.step(sim.initial_state(), 0)
        self.assertEqual(first.following, frozenset({"test:7"}))
        self.assertFalse(first.berries)
        self.assertEqual(first.collect_delay, 17)
        early = sim.roll(first, 0, 20)
        self.assertFalse(early.berries)
        result = sim.roll(early, 0, 20)
        self.assertEqual(result.berries, frozenset({"test:7"}))
        self.assertFalse(result.following)

    def test_berry_delay_expires_before_first_collection_tick(self):
        berry = Entity("strawberry", 7, 24, 159, 0, 0)
        sim = Simulator(flat_map([berry]))
        picked = sim.step(sim.initial_state(), 0)
        waiting = sim.roll(picked, 0, 16)
        self.assertEqual((waiting.collect_delay, waiting.collect_ground), (1, 0))
        elapsed = sim.step(waiting, 0)
        self.assertEqual((elapsed.collect_delay, elapsed.collect_ground), (0, 0))
        collecting = sim.step(elapsed, 0)
        self.assertEqual((collecting.collect_delay, collecting.collect_ground), (0, 1))

    def test_berry_commits_on_ninth_safe_ground_tick(self):
        sim = Simulator(flat_map())
        carrying = replace(sim.initial_state(), following=frozenset({"test:7"}),
                           follow_order=("test:7",), collect_delay=0)
        before = sim.roll(carrying, 0, 8)
        self.assertEqual(before.collect_ground, 8)
        self.assertFalse(before.berries)
        committed = sim.step(before, 0)
        self.assertEqual(committed.berries, frozenset({"test:7"}))
        self.assertFalse(committed.following)
        self.assertEqual(committed.collect_ground, 0)

    def test_branching_is_immutable_and_repeatable(self):
        sim = Simulator(flat_map())
        start = sim.initial_state()
        left = sim.roll(start, LEFT | JUMP, 12)
        right = sim.roll(start, RIGHT | JUMP, 12)
        self.assertEqual(start.x, 24)
        self.assertLess(left.x, right.x)
        self.assertEqual(right, sim.roll(start, RIGHT | JUMP, 12))
        self.assertEqual(right, SimState.from_dict(right.to_dict()))

    def test_jumpthru_only_blocks_downward_crossing(self):
        platform = Entity("jumpThru", 1, 16, 136, 40, 8)
        sim = Simulator(flat_map([platform]))
        rising = replace(sim.initial_state(), x=32, y=151, vy=-150, on_ground=False)
        self.assertLess(sim.roll(rising, JUMP, 12).y, 136)
        falling = replace(sim.initial_state(), x=32, y=120, vy=160, on_ground=False)
        self.assertEqual(sim.roll(falling, 0, 20).y, 136)

    def test_unsupported_is_explicit_and_special_berries_not_completed(self):
        unsupported = Entity("dreamBlock", 1, 40, 80, 32, 16)
        gold = Entity("goldenBerry", 2, 24, 159, 0, 0)
        sim = Simulator(flat_map([unsupported, gold]))
        report = sim.capability_report("test")
        self.assertFalse(report["planning_supported"])
        self.assertIn("dreamBlock", report["unsupported"])
        self.assertIn("goldenBerry", report["unsupported_collectibles"])
        self.assertFalse(sim.roll(sim.initial_state(), 0, 60).berries)

    def test_winged_berry_is_not_granted_after_dash(self):
        berry = Entity("strawberry", 3, 100, 159, 0, 0, {"winged": True})
        sim = Simulator(flat_map([berry]))
        result = sim.roll(sim.initial_state(), RIGHT | DASH, 60)
        self.assertIn("test:3", result.fled)
        self.assertNotIn("test:3", result.berries | result.following)

    def test_dash_does_not_flee_caught_or_collected_winged_berries(self):
        berries = (Entity("strawberry", 3, 100, 159, 0, 0, {"winged": True}),
                   Entity("strawberry", 4, 120, 159, 0, 0, {"winged": True}),
                   Entity("strawberry", 5, 140, 159, 0, 0, {"winged": True}))
        sim = Simulator(flat_map(berries))
        start = replace(sim.initial_state(), following=frozenset({"test:3"}),
                        follow_order=("test:3",), berries=frozenset({"test:4"}), collect_delay=18)
        result = sim.roll(start, RIGHT | DASH, 2)
        self.assertEqual(result.fled, frozenset({"test:5"}))
        self.assertIn("test:3", result.following)
        self.assertIn("test:4", result.berries)

    def test_adjacent_room_transition(self):
        base = flat_map()
        r = base.rooms["test"]
        other = replace(r, name="next", x=320, spawns=((344.0,168.0),))
        sim = Simulator(MapData("fixture", {r.name:r, other.name:other}, Path("fixture.bin")))
        start = replace(sim.initial_state(), x=315, vx=90)
        result = sim.roll(start, RIGHT, 4)
        self.assertEqual(result.room, "next")
        self.assertFalse(result.dead)

    def test_up_transition_uses_runtime_180_height_strict_padding_and_dash_delay(self):
        base = flat_map()
        below = base.rooms["test"]
        above = replace(below, name="above", y=-184, solids=(), spawns=((24.0, -16.0),))
        sim = Simulator(MapData("fixture", {below.name: below, above.name: above}, Path("fixture.bin")))
        entered = sim._bounds(replace(sim.initial_state(), x=24, y=4, vy=-105, on_ground=False))
        self.assertEqual((entered.room, entered.y), ("above", -9))
        self.assertEqual((entered.vx, entered.vy, entered.var_jump, entered.dash_cooldown), (0, -105, 12, 12))
        self.assertTrue(entered.auto_jump_active)
        early_dash = sim.step(entered, UP | DASH)
        self.assertEqual(early_dash.state, 0)
        self.assertEqual(early_dash.dashes, 1)

    def test_left_transition_uses_strict_right_edge_padding(self):
        base = flat_map()
        right = base.rooms["test"]
        left = replace(right, name="left", x=-320, spawns=((-24.0,168.0),))
        sim = Simulator(MapData("fixture", {right.name: right, left.name: left}, Path("fixture.bin")))
        entered = sim._bounds(replace(sim.initial_state(), x=3, vx=-90))
        self.assertEqual((entered.room, entered.x), ("left", -5))

    def test_reentering_room_restores_departed_uncollected_winged_berry(self):
        base = flat_map()
        right = base.rooms["test"]
        left = replace(right, name="left", x=-320, spawns=((-24.0,168.0),))
        sim = Simulator(MapData("fixture", {right.name: right, left.name: left}, Path("fixture.bin")))
        start = replace(sim.initial_state(), x=3, fled=frozenset({"left:4", "test:5"}))
        entered = sim._bounds(start)
        self.assertEqual(entered.fled, frozenset({"test:5"}))

    def test_dash_block_breaks_only_with_dash_and_rebounds(self):
        block = Entity("dashBlock", 9, 64, 128, 16, 40)
        sim = Simulator(flat_map([block]))
        walking = sim.roll(sim.initial_state(), RIGHT, 60)
        self.assertLessEqual(walking.x, 60)
        self.assertFalse(walking.broken)
        dashed = sim.roll(replace(sim.initial_state(), x=48), RIGHT | DASH, 8)
        self.assertIn("test:9", dashed.broken)
        self.assertLess(dashed.vx, 0)
        self.assertLess(dashed.vy, 0)
        self.assertFalse(sim.initial_state().broken)

    def test_dash_block_uses_attack_window_and_matching_collision_direction(self):
        block = Entity("dashBlock", 9, 64, 128, 16, 40)
        sim = Simulator(flat_map([block]))
        after_dash = replace(sim.initial_state(), x=60, y=150, vx=160, state=0,
                             dash_attack=3, dash_dx=1, dashes=0, on_ground=False, wall_slide=12)
        rebound = sim.step(after_dash, RIGHT)
        self.assertIn("test:9", rebound.broken)
        self.assertEqual((rebound.vx, rebound.vy), (-120, -120))
        self.assertEqual((rebound.dash_attack, rebound.wall_slide), (0, 72))
        wrong_axis = sim.step(replace(after_dash, dash_dx=0, dash_dy=-1), RIGHT)
        self.assertFalse(wrong_axis.broken)
        expired = sim.step(replace(after_dash, dash_attack=1), RIGHT)
        self.assertFalse(expired.broken)

    def test_crumble_platform_falls_and_never_confirms_berries(self):
        platform = Entity("crumbleBlock", 4, 8, 120, 64, 8)
        berry = Entity("strawberry", 7, 24, 111, 0, 0)
        sim = Simulator(flat_map([platform, berry]))
        start = replace(sim.initial_state(), y=120)
        waiting = sim.roll(start, 0, 32)
        self.assertEqual(waiting.y, 120)
        self.assertFalse(waiting.berries)
        falling = sim.roll(waiting, 0, 20)
        self.assertGreater(falling.y, 120)
        self.assertEqual(dict((row[0], row[1]) for row in falling.mechanisms)["test:4"], 3)

    def test_zip_mover_carries_rider_and_state_is_branch_local(self):
        platform = Entity("zipMover", 5, 8, 120, 64, 16, {}, ((120.0, 120.0),))
        sim = Simulator(flat_map([platform]))
        start = replace(sim.initial_state(), y=120)
        carried = sim.roll(start, 0, 38)
        self.assertGreater(carried.x, start.x + 90)
        self.assertEqual(carried.y, 120)
        self.assertEqual(carried, sim.roll(start, 0, 38))
        self.assertFalse(start.mechanisms)
        self.assertIn("zipMover", sim.capability_report()["approximate_entities"])

    def test_falling_block_lands_on_tiles_and_carries_player(self):
        platform = Entity("fallingBlock", 8, 8, 100, 64, 16)
        sim = Simulator(flat_map([platform]))
        start = replace(sim.initial_state(), y=100)
        landed = sim.roll(start, 0, 100)
        self.assertFalse(landed.dead)
        self.assertEqual(landed.y, 152)
        self.assertEqual(dict((row[0], row[1]) for row in landed.mechanisms)["test:8"], 3)

    def test_default_spawn_is_nearest_to_room_bottom_left(self):
        base = flat_map()
        original = base.rooms["test"]
        room = replace(original, spawns=((290.0, 168.0), (40.0, 80.0), (24.0, 168.0)))
        sim = Simulator(MapData("fixture", {room.name: room}, Path("fixture.bin")))
        self.assertEqual((sim.initial_state().x, sim.initial_state().y), (24.0, 168.0))
        self.assertEqual(sim.initial_state(spawn_index=0).x, 290)

    def test_zip_attached_spikes_move_with_support(self):
        platform = Entity("zipMover", 5, 32, 120, 32, 16, {}, ((128.0, 120.0),))
        spike = Entity("spikesUp", 6, 40, 120, 16, 0)
        sim = Simulator(flat_map([platform, spike]))
        moved = replace(sim.initial_state(), x=48, y=120,
                        mechanisms=(("test:5", 3, 30, 128.0, 120.0, 0.0),))
        self.assertFalse(sim._hazard_hit(moved))
        self.assertTrue(sim._hazard_hit(replace(moved, x=144)))
        report = sim.capability_report()
        self.assertTrue(report["planning_supported"])
        self.assertEqual(report["approximate_entities"]["platform_attached_spikes"], 1)

    def test_nearby_spike_is_not_attached_without_source_contact_predicate(self):
        platform = Entity("zipMover", 5, 32, 120, 32, 16, {}, ((128.0, 120.0),))
        spike = Entity("spikesUp", 6, 40, 115, 16, 0)
        sim = Simulator(flat_map([platform, spike]))
        moved = replace(sim.initial_state(), x=48, y=115,
                        mechanisms=(("test:5", 3, 30, 128.0, 120.0, 0.0),))
        self.assertTrue(sim._hazard_hit(moved))
        self.assertFalse(sim._hazard_hit(replace(moved, x=144)))

    def test_destroyed_falling_block_removes_attached_spikes(self):
        platform = Entity("fallingBlock", 5, 32, 120, 32, 16)
        spike = Entity("spikesLeft", 6, 32, 120, 0, 16)
        sim = Simulator(flat_map([platform, spike]))
        near = replace(sim.initial_state(), x=29, y=131)
        self.assertTrue(sim._hazard_hit(near))
        gone = replace(near, mechanisms=(("test:5", 4, 0, 32.0, 210.0, 160.0),))
        self.assertFalse(sim._hazard_hit(gone))

    def test_indefinite_autojump_keeps_half_gravity_after_variable_jump(self):
        sim = Simulator(flat_map())
        apex = replace(sim.initial_state(), y=90, vy=-30, on_ground=False,
                       grace=0, var_jump=0, auto_jump=0, auto_jump_active=True)
        next_state = sim.step(apex, 0)
        self.assertAlmostEqual(next_state.vy, -22.5, delta=1e-4)  # Half gravity; bridge Single delta is .0166667f.
        self.assertTrue(next_state.auto_jump_active)
        self.assertEqual(next_state.auto_jump, 0)
        after_apex = sim.roll(apex, 0, 10)
        self.assertTrue(after_apex.auto_jump_active)
        self.assertLess(after_apex.vy, sim.roll(replace(apex, auto_jump_active=False), 0, 10).vy)

    def test_positive_autojump_timer_expires_but_zero_timer_does_not(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(), y=90, vy=0, on_ground=False,
                        grace=0, auto_jump=2, auto_jump_active=True)
        first = sim.step(start, 0)
        self.assertTrue(first.auto_jump_active)
        self.assertEqual(first.auto_jump, 1)
        second = sim.step(first, 0)
        self.assertFalse(second.auto_jump_active)
        self.assertEqual(second.auto_jump, 0)
        self.assertAlmostEqual(second.vy, 22.5, delta=1e-4)  # Half gravity first tick, full on expiry.

    def test_landing_and_manual_jump_clear_autojump(self):
        sim = Simulator(flat_map())
        grounded = replace(sim.initial_state(), auto_jump_active=True, auto_jump=5)
        self.assertFalse(sim.step(grounded, 0).auto_jump_active)
        jumped = sim.step(grounded, JUMP)
        self.assertFalse(jumped.auto_jump_active)
        self.assertEqual(jumped.auto_jump, 0)

    def test_dash_exit_sets_indefinite_autojump(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(), y=110, on_ground=False, grace=0)
        after_dash = sim.roll(start, RIGHT | DASH, 12)
        self.assertEqual(after_dash.state, 0)
        self.assertTrue(after_dash.auto_jump_active)
        self.assertEqual(after_dash.auto_jump, 0)

    def test_floor_spring_uses_superbounce_indefinite_flag_and_top(self):
        spring = Entity("spring", 5, 24, 168, 0, 0)
        sim = Simulator(flat_map([spring]))
        hit = sim.step(replace(sim.initial_state(), vx=80), 0)
        self.assertEqual(hit.vy, -185)
        self.assertEqual(hit.vx, 0)
        self.assertEqual(hit.y, 162)  # Spring.Top = position.Y - 6.
        self.assertTrue(hit.auto_jump_active)
        self.assertEqual(hit.auto_jump, 0)

    def test_player_can_use_disables_all_spring_orientations_without_refill(self):
        for kind, x in (("spring", 64), ("wallSpringLeft", 68), ("wallSpringRight", 60)):
            for value in (False, "false", " FALSE "):
                with self.subTest(kind=kind, value=value):
                    spring = Entity(kind, 5, 64, 128, 0, 0, {"playerCanUse": value})
                    sim = Simulator(flat_map([spring]))
                    player = replace(sim.initial_state(), x=x, y=128, vx=0, vy=0,
                                     dashes=0, stamina=10, on_ground=False, grace=0)
                    self.assertEqual(sim._entities(player), player)

    def test_spring_flag_true_missing_and_invalid_use_native_true_default(self):
        for kind, x in (("spring", 64), ("wallSpringLeft", 68), ("wallSpringRight", 60)):
            for attrs in ({}, {"playerCanUse": True}, {"playerCanUse": " TRUE "},
                          {"playerCanUse": 0}, {"playerCanUse": "unrecognized"}):
                with self.subTest(kind=kind, attrs=attrs):
                    spring = Entity(kind, 5, 64, 128, 0, 0, attrs)
                    sim = Simulator(flat_map([spring]))
                    player = replace(sim.initial_state(), x=x, y=128, vx=0, vy=0,
                                     dashes=0, stamina=10, on_ground=False, grace=0)
                    bounced = sim._entities(player)
                    self.assertEqual((bounced.dashes, bounced.stamina), (1, 110))
                    self.assertEqual(bounced.vy, -185 if kind == "spring" else -140)
                    self.assertEqual(bounced.vx, 0 if kind == "spring" else (240 if kind == "wallSpringLeft" else -240))

    def test_dash_coroutine_has_ten_full_speed_moves_before_exit(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(), y=110, on_ground=False, grace=0)
        trace = sim.trace(start, [RIGHT | DASH] * 12)
        self.assertEqual(trace[1].x, start.x)  # Initial yield-null update.
        self.assertEqual([s.vx for s in trace[2:12]], [240.0] * 10)
        self.assertEqual(trace[11].x - trace[1].x, 40)
        self.assertEqual(trace[11].state, 2)
        self.assertEqual((trace[12].state, trace[12].vx), (0, 160.0))

    def test_dash_update_jump_precedes_coroutine_direction_assignment(self):
        sim = Simulator(flat_map())
        startup = sim.step(sim.initial_state(), UP | DASH)
        jumped = sim.step(startup, UP | JUMP)
        self.assertEqual(jumped.state, 0)
        self.assertEqual((jumped.vx, jumped.vy), (260, -105))
        self.assertEqual(jumped.dash_attack, 0)

    def test_entity_contact_uses_hurtbox_above_feet(self):
        berry = Entity("strawberry", 7, 24, 153, 0, 0)
        spring = Entity("spring", 8, 64, 168, 0, 0)
        sim = Simulator(flat_map([berry, spring]))
        # The body reaches the berry/spring; its bottom two pixels are absent
        # from the hurtbox used by PlayerCollider.Check.
        near_berry = replace(sim.initial_state(), y=147, on_ground=False)
        self.assertFalse(sim._entities(near_berry).following)
        self.assertIn("test:7", sim._entities(replace(near_berry, y=149)).following)
        near_spring = replace(near_berry, x=64, y=164)
        self.assertEqual(sim._entities(near_spring).vy, 0)
        self.assertEqual(sim._entities(replace(near_spring, y=165)).vy, -185)

    def test_corner_correction_clears_actor_fractional_remainder(self):
        wall = Entity("solid", 1, 64, 128, 16, 40)
        sim = Simulator(flat_map([wall]))
        start = replace(sim.initial_state(), x=59, y=130, vx=240, vy=0,
                        on_ground=False, grace=0, state=2, dash_frames=5,
                        dash_dx=1, dash_attack=12, rem_x=0.3)
        moved = sim.step(start, RIGHT)
        self.assertEqual(moved.y, 128)
        self.assertEqual(moved.vx, 240)
        self.assertEqual(moved.rem_x, 0)

    def test_ceiling_clears_aged_nine_tick_jump_timer_on_native_float_boundary(self):
        ceiling = Entity("solid", 1, 0, 64, 320, 16)
        sim = Simulator(flat_map([ceiling]))
        start = replace(sim.initial_state(), x=100, y=91, on_ground=False, grace=0,
                        vy=0, var_jump=10, var_speed=-105, prev_input=JUMP)
        hit = sim.step(start, JUMP)
        self.assertEqual((hit.y, hit.vy, hit.var_jump), (91, 0, 0))
        falling = sim.step(hit, JUMP)
        self.assertAlmostEqual(falling.vy, 7.500015, places=4)
        self.assertAlmostEqual(falling.rem_y, .125000492, places=6)
        younger = sim.step(replace(start, var_jump=11), JUMP)
        self.assertEqual(younger.var_jump, 10)

    def test_same_frame_dash_block_rebound_retains_exact_point15_jump_timer(self):
        ceiling = Entity("solid", 1, 0, 64, 320, 16)
        block = Entity("dashBlock", 2, 104, 88, 8, 80)
        sim = Simulator(flat_map([ceiling, block]))
        start = replace(sim.initial_state(), x=99, y=91, vx=240, vy=0,
                        on_ground=False, grace=0, state=2, dash_frames=5,
                        dash_dx=1, dash_attack=12, dashes=0)
        rebounded = sim.step(start, RIGHT)
        self.assertIn("test:2", rebounded.broken)
        self.assertEqual((rebounded.y, rebounded.vy, rebounded.var_jump), (91, 0, 9))
        self.assertTrue(rebounded.auto_jump_active)
        self.assertEqual(sim.step(rebounded, 0).var_jump, 0)

    def test_wall_retention_restores_speed_after_clearing_lip(self):
        wall = Entity("solid", 1, 64, 136, 16, 32)
        sim = Simulator(flat_map([wall]))
        cleared = replace(sim.initial_state(), x=60, y=130, on_ground=False,
                          wall_retention=3, wall_retained=180)
        restored = sim.step(cleared, RIGHT)
        self.assertGreater(restored.vx, 170)
        self.assertEqual(restored.wall_retention, 0)
        reversing = sim.step(replace(cleared, vx=-1), LEFT)
        self.assertLess(reversing.vx, 0)
        self.assertEqual(reversing.wall_retention, 0)

    def test_runtime_scripts_do_not_hide_unsupported_cassette_solids(self):
        scripts = [Entity(kind, n, 40, 80, 16, 16) for n, kind in enumerate(
            ("npc", "eventTrigger", "goldenBerryCollectTrigger"))]
        report = Simulator(flat_map(scripts)).capability_report()
        self.assertTrue(report["planning_supported"])
        self.assertIn("eventTrigger:runtime_script", report["approximate_entities"])
        report = Simulator(flat_map(scripts + [Entity("cassetteBlock", 9, 80, 80, 16, 16)])).capability_report()
        self.assertFalse(report["planning_supported"])
        self.assertIn("cassetteBlock", report["unsupported"])

    def test_grab_enters_climb_before_last_buffered_jump_and_slips_at_lip(self):
        wall = Entity("solid", 1, 64, 128, 16, 40)
        sim = Simulator(flat_map([wall]))
        mask = RIGHT | GRAB | JUMP
        start = replace(sim.initial_state(), x=60, y=129, vx=43.3333, vy=102.64756,
                        on_ground=False, grace=0, jump_buffer=2, prev_input=mask)
        grabbed = sim.step(start, mask)
        self.assertEqual(grabbed.state, 1)
        self.assertAlmostEqual(grabbed.vy, 20.529512, delta=1e-5)
        self.assertEqual(grabbed.jump_buffer, 1)
        self.assertEqual(grabbed.climb_wait, 6)
        slipped = sim.step(grabbed, mask)
        self.assertEqual(slipped.state, 1)
        self.assertEqual(slipped.vy, 30)
        self.assertEqual(slipped.jump_buffer, 0)

    def test_climb_jump_retains_its_new_speed_after_clearing_lip(self):
        wall = Entity("solid", 1, 64, 128, 16, 40)
        sim = Simulator(flat_map([wall]))
        mask = RIGHT | GRAB | JUMP
        start = replace(sim.initial_state(), x=60, y=129, state=1, vx=0, vy=8.5,
                        on_ground=False, grace=0, jump_buffer=5, prev_input=mask,
                        wall_retention=3, wall_retained=90)
        jumped = sim.step(start, mask)
        self.assertEqual(jumped.state, 0)
        self.assertEqual(jumped.wall_retained, 40)
        self.assertEqual(jumped.vx, 0)  # Horizontal collision before rising.
        cleared = sim.step(jumped, mask)
        self.assertAlmostEqual(cleared.vx, 40 + 1000 * 0.65 / 60, delta=1e-4)

    def test_single_precision_flight_does_not_accept_one_pixel_spike_miss(self):
        # Source Single arithmetic plus the bridge's .0166667f delta places the
        # final body at x=802. Exact-1/60 doubles used to stop at x=801 and falsely
        # accept this flight. The trajectory was also present in ordinary live
        # execution telemetry; reproducing it here never opens the game.
        spike = Entity("spikesLeft", 1, 808, 936, 0, 32)
        sim = Simulator(flat_map([spike], width=140, height=160))
        start = replace(sim.initial_state(), x=776, y=1000, on_ground=False, grace=0)
        actions = [UP | DASH] * 7 + [RIGHT | JUMP] * 22 + [0] * 8
        end = sim.trace(start, actions)[-1]
        self.assertEqual((end.x, end.y), (802, 971))
        self.assertTrue(end.dead)
        self.assertEqual(end.reason, "hazard")

    def test_up_climb_hops_when_hands_clear_before_lower_body(self):
        wall = Entity("solid", 1, 64, 128, 32, 40)
        sim = Simulator(flat_map([wall]))
        start = replace(sim.initial_state(), x=60, y=134, state=1, vy=-15,
                        on_ground=False, grace=0, climb_wait=0)
        hopped = sim.step(start, GRAB | UP)
        self.assertEqual((hopped.state, hopped.vx, hopped.vy), (0, 0, -120))
        self.assertEqual((hopped.hop_wait_x, hopped.force_x, hopped.force_frames), (1, 0, 12))
        self.assertEqual(hopped.x, 60)  # Body still touches wall; wait to move right.
        cleared = sim.roll(hopped, GRAB | UP, 4)
        self.assertEqual(cleared.hop_wait_x, 0)
        self.assertGreater(cleared.x, 60)
        self.assertAlmostEqual(cleared.vx, 89.16665, delta=1e-4)

    def test_spike_ledge_blocker_stops_climb_before_unsafe_automatic_hop(self):
        # Live room6 failure reproduced at a translated static ledge: the real
        # player stopped seven pixels below the spike-top but the old model
        # climbed through and invented a hop. These are offline-only inputs.
        wall = Entity("solid",1,64,128,32,40)
        spike = Entity("spikesUp",2,64,128,32,0)
        sim = Simulator(flat_map([wall,spike]))
        start = replace(sim.initial_state(),x=60,y=135,state=1,vy=-45,
                        stamina=106.969696,rem_y=.277990878,prev_input=GRAB|UP,
                        on_ground=False,grace=0,climb_wait=0)
        stopped = sim.step(start,GRAB|UP)
        self.assertEqual((stopped.x,stopped.y,stopped.vy,stopped.state),(60,135,0,1))
        self.assertAlmostEqual(stopped.stamina,106.803032,delta=1e-5)
        self.assertAlmostEqual(stopped.rem_y,start.rem_y,delta=1e-7)
        held = sim.roll(stopped,GRAB|UP,12)
        self.assertEqual((held.x,held.y,held.state),(60,135,1))
        self.assertFalse(held.dead)
        self.assertEqual(held.hop_wait_x,0)

    def test_spikes_elsewhere_do_not_disable_safe_climb_hop(self):
        wall = Entity("solid",1,64,128,32,40)
        distant = Entity("spikesUp",2,88,128,8,0)
        sim = Simulator(flat_map([wall,distant]))
        start = replace(sim.initial_state(),x=60,y=134,state=1,vy=-15,
                        on_ground=False,grace=0,climb_wait=0)
        hopped = sim.step(start,GRAB|UP)
        self.assertEqual((hopped.state,hopped.vy),(0,-120))
        self.assertEqual(hopped.hop_wait_x,1)

    def test_downward_spikes_have_no_native_ledge_blocker(self):
        down = Entity("spikesDown",1,64,128,32,0)
        sim = Simulator(flat_map([down]))
        start = replace(sim.initial_state(),x=60,y=135,on_ground=False)
        self.assertFalse(sim._ledge_blocked(start,68,135))

    def test_climb_begin_clears_wall_retention_from_previous_collision(self):
        wall = Entity("solid",1,64,128,32,40)
        sim = Simulator(flat_map([wall]))
        start = replace(sim.initial_state(),x=60,y=144,vy=145,vx=0,
                        on_ground=False,grace=0,wall_retention=4,wall_retained=90)
        grabbed = sim.step(start,GRAB|UP)
        self.assertEqual(grabbed.state,1)
        self.assertEqual(grabbed.wall_retention,0)

    def test_releasing_down_before_dash_jump_keeps_hyperjump_multiplier(self):
        # Actual request15 used a crouched down-right dash then right+jump.
        # Native DashUpdate reads Ducking before SuperJump clears it, yielding
        # 325 horizontal / -52.5 vertical speed rather than 260 / -105.
        sim = Simulator(flat_map())
        duck_dash = sim.roll(sim.initial_state(),RIGHT|DOWN|DASH,7)
        self.assertTrue(duck_dash.ducking)
        self.assertEqual(duck_dash.state,2)
        hyper = sim.step(duck_dash,RIGHT|JUMP)
        self.assertEqual((hyper.vx,hyper.vy,hyper.state),(325,-52.5,0))
        self.assertFalse(hyper.ducking)

    def test_airborne_down_dash_landing_keeps_ducking_until_wave_jump(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(),y=160,on_ground=False,grace=0)
        landed = sim.roll(start,RIGHT|DOWN|DASH,7)
        self.assertTrue(landed.on_ground)
        self.assertTrue(landed.ducking)
        released_down = sim.step(landed,RIGHT)
        self.assertEqual(released_down.state,2)
        self.assertTrue(released_down.ducking)
        wavedash = sim.step(released_down,RIGHT|JUMP)
        self.assertEqual((wavedash.vx,wavedash.vy),(325,-52.5))

    def test_spiked_jumpthrough_blocks_upward_passage_boost(self):
        platform = Entity("jumpThru",1,32,136,32,8)
        spikes = Entity("spikesUp",2,32,136,32,0)
        safe = Simulator(flat_map([platform]))
        dangerous = Simulator(flat_map([platform,spikes]))
        start = replace(safe.initial_state(),x=30,y=139,vx=90,vy=-45,
                        rem_y=.237810135,on_ground=False,grace=0)
        boosted = safe.step(start,RIGHT)
        blocked = dangerous.step(start,RIGHT)
        self.assertEqual(boosted.y,138)
        self.assertEqual(blocked.y,139)
        self.assertAlmostEqual(blocked.vy,boosted.vy,delta=1e-5)
        self.assertFalse(blocked.dead)  # Still rising; spike direction matters.

    def test_up_spike_uses_hurtbox_bottom_at_jumpthrough_apex(self):
        platform = Entity("jumpThru",1,32,136,32,8)
        spikes = Entity("spikesUp",2,32,136,32,0)
        sim = Simulator(flat_map([platform,spikes]))
        start = replace(sim.initial_state(),x=40,y=138,vy=-15,
                        on_ground=False,grace=0)
        self.assertFalse(sim._hazard_hit(start))
        apex = sim.step(start,0)
        self.assertEqual(apex.y,138)
        self.assertTrue(apex.dead)
        self.assertEqual(apex.reason,"hazard")

    def test_airborne_down_dash_crouches_initial_frame_then_unducks_when_falling(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(),y=100,on_ground=False,grace=0)
        beginning = sim.step(start,LEFT|DOWN|DASH)
        self.assertEqual((beginning.state,beginning.vy,beginning.ducking),(2,0,True))
        moving = sim.step(beginning,LEFT|DOWN|DASH)
        self.assertGreater(moving.vy,0)
        self.assertFalse(moving.ducking)

    def test_returning_zip_carries_new_landing_same_frame_with_fractional_lift(self):
        mover = Entity("zipMover",2,2096,-1152,32,16,{},((2160.,-1152.),))
        room = Room("test",2008,-1384,320,288,("0"*40,)*36,(mover,),((2169.,-1154.),))
        sim = Simulator(MapData("zip-live",{"test":room},Path("zip-live.bin")))
        start = SimState("test",2169,-1154,vy=160,
                         mechanisms=(("test:2",4,39,2151.839599609375,-1152.,0.),))
        landed = sim.step(start,0)
        self.assertEqual((landed.x,landed.y),(2168,-1152))
        self.assertAlmostEqual(landed.lift_vx,-24.8437023,delta=1e-4)
        self.assertEqual(landed.lift_timer,10)
        self.assertNotEqual(landed.lift_vx,-60)  # Pixel carry is not the lift boost.

    def test_lift_speed_expires_after_native_grace_interval(self):
        sim = Simulator(flat_map())
        start = replace(sim.initial_state(),lift_vx=-26.5429173,lift_timer=10)
        waiting = sim.roll(start,0,9)
        self.assertEqual(waiting.lift_timer,1)
        self.assertEqual(waiting.lift_vx,start.lift_vx)
        expired = sim.step(waiting,0)
        self.assertEqual((expired.lift_timer,expired.lift_vx,expired.lift_vy),(0,0,0))

    def test_vertical_dash_uses_fifth_corner_correction_pixel(self):
        ceiling = Entity("solid",1,64,80,64,8)
        sim = Simulator(flat_map([ceiling]))
        start = replace(sim.initial_state(),x=65,y=100,vy=-240,state=2,
                        dash_frames=5,dash_attack=10,dash_dx=0,dash_dy=-1,
                        on_ground=False,grace=0)
        escaped = sim.step(start,0)
        self.assertEqual((escaped.x,escaped.y,escaped.vy),(60,98,-240))
        ordinary = sim.step(replace(start,state=0,dash_frames=0,dash_attack=0),0)
        self.assertEqual((ordinary.x,ordinary.vy),(65,0))

    def test_climb_hand_point_at_large_tile_boundary_is_solid(self):
        # x+epsilon-epsilon can round below x2288 and create an empty tile
        # rectangle. Native SlipCheck uses an exact point, so this is a wall.
        rows=("0"*40,)*9+("0"*35+"1"*5,)*27
        room = Room("test",2008,-1384,320,288,rows,(),((2284.,-1288.),))
        sim = Simulator(MapData("large-boundary",{"test":room},Path("large.bin")))
        start=SimState("test",2284,-1288,state=1,vy=.0000425338767,
                       climb_wait=6,stamina=27.5,rem_y=-.458375663)
        held=sim.step(start,GRAB|UP)
        self.assertEqual((held.y,held.vy,held.state),(-1288,0,1))
        climbing=sim.roll(held,GRAB|UP,5)
        self.assertEqual(climbing.state,1)
        self.assertLess(climbing.vy,0)


if __name__ == "__main__":
    unittest.main()
