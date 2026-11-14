"""Reference-derived cassette mechanics; no game, guessed phase, or time sleeps."""
from dataclasses import replace
import json
import struct
import unittest

from celeste_search.cassette import (CassetteActor, CassetteBlockState, CassetteClock,
                                    advance_clock, step_cassette)


def clock(**changes):
    return replace(CassetteClock(0, 0.0, 0, 1.0, 0, 2, True, 5, True, True, 0), **changes)


def block(identity="a", **changes):
    return replace(CassetteBlockState(identity, 0, 1.0, 0.0, 20.0, 16.0, 8.0,
                                     True, True, True, 2, True, (identity,), 2,
                                     0.0, 0.0, 0.0), **changes)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class CassetteTests(unittest.TestCase):
    def test_strict_telemetry_roundtrip_hash_and_unknown_rejection(self):
        c, b = clock(beat_timer=f32(.123), beat_index=42), block(group_ids=("a",))
        self.assertEqual(CassetteClock.from_dict(json.loads(json.dumps(c.to_dict()))), c)
        self.assertEqual(CassetteBlockState.from_dict(json.loads(json.dumps(b.to_dict()))), b)
        self.assertEqual(hash(CassetteBlockState.from_dict(b.to_dict())), hash(b))
        for key in ("beat_timer", "music_ready", "current_index", "update_order"):
            data = c.to_dict()
            del data[key]
            with self.assertRaises(ValueError):
                CassetteClock.from_dict(data)
        for key in ("collidable", "group_ids", "activated", "rem_y"):
            data = b.to_dict()
            del data[key]
            with self.assertRaises(ValueError):
                CassetteBlockState.from_dict(data)
        for change in ({"present": False}, {"beat_timer": float("nan")},
                       {"active": 1}, {"unavailable": ["private phase"]}):
            with self.assertRaises(ValueError):
                CassetteClock.from_dict({**c.to_dict(), **change})

    def test_manager_single_precision_beat_and_pre_toggle_sequence(self):
        c = clock()
        events = []
        for frame in range(1, 161):
            c, event = advance_clock(c)
            if event:
                events.append((frame, event))
        self.assertEqual(events, [(70, ("will_toggle", 1)), (80, ("activate", 1)),
                                  (150, ("will_toggle", 0)), (160, ("activate", 0))])
        # Native only advances one sixteenth per Update, even with excessive dt.
        high, _ = advance_clock(clock(), 1.0)
        self.assertEqual(high.beat_index, 1)
        self.assertEqual(high.beat_timer, f32(1.0 - f32(1 / 6)))

    def test_music_start_lead_in_and_inactive_clock_do_not_invent_phase(self):
        c = clock(is_level_music=False, music_ready=False, lead_beats=1, beat_index=7,
                  beat_timer=f32(1 / 6 - 1 / 60))
        started, event = advance_clock(c)
        self.assertTrue(started.music_ready)
        self.assertEqual(started.beat_timer, c.beat_timer)
        self.assertIsNone(event)
        advanced, event = advance_clock(started)
        self.assertEqual(event, ("activate", 1))
        self.assertEqual((advanced.lead_beats, advanced.beat_index), (0, 0))
        inactive = replace(c, active=False)
        self.assertEqual(advance_clock(inactive), (inactive, None))

    def test_pre_toggle_is_real_position_change_and_carries_rider(self):
        c = clock(beat_index=6, beat_timer=f32(1 / 6 - 1 / 60))
        active = block()
        incoming = block("b", x=32, index=1, activated=False, collidable=False,
                         y=22, block_height=0, update_order=3)
        actor = CassetteActor(8, 20, rem_y=.375)
        result = step_cassette(c, (active, incoming), actor)
        self.assertEqual([b.y for b in result.blocks], [21, 21])
        self.assertEqual([b.block_height for b in result.blocks], [1, 1])
        self.assertEqual(result.actor.y, 21)
        self.assertEqual(result.actor.rem_y, .375)
        self.assertAlmostEqual(result.actor.lift_vy, f32(1 / f32(1 / 60)))
        self.assertEqual(result.actor.lift_timer, 10)
        self.assertEqual(actor.y, 20)  # no mutation of a search branch.

    def test_activation_wiggles_up_then_moves_the_real_solid_one_pixel(self):
        b = block(y=21, collidable=False, block_height=1)
        actor = CassetteActor(8, 23, rem_y=.125)
        result = step_cassette(clock(active=False), (b,), actor)
        self.assertTrue(result.blocks[0].collidable)
        self.assertEqual(result.blocks[0].y, 20)
        self.assertEqual(result.actor.y, 20)  # wiggle 2, then Solid.MoveV(-1).
        self.assertEqual(result.actor.rem_y, .125)
        self.assertIn(("wiggle_up", -2), [(e.kind, e.amount) for e in result.events])
        self.assertLess(result.actor.lift_vy, -59)

    def test_obstruction_defers_whole_group_and_retries_without_new_beat(self):
        group = ("a", "b")
        a = block(y=21, activated=True, collidable=False, block_height=1, group_ids=group)
        b = block("b", x=16, y=21, activated=True, collidable=False, block_height=1,
                  group_ids=group, group_leader=False, update_order=3)
        actor = CassetteActor(8, 23)
        ceiling = ((-8, 5, 40, 12),)
        blocked = step_cassette(clock(active=False), (a, b), actor, solid_rects=ceiling)
        self.assertFalse(any(item.collidable for item in blocked.blocks))
        self.assertEqual(blocked.actor, actor)
        clear = step_cassette(blocked.clock, blocked.blocks, CassetteActor(48, 10))
        self.assertTrue(all(item.collidable for item in clear.blocks))
        self.assertEqual([item.y for item in clear.blocks], [20, 20])

    def test_group_four_pixel_overlap_prohibits_individual_wiggle(self):
        group = ("a", "b")
        a = block(y=21, collidable=False, block_height=1, group_ids=group)
        b = block("b", x=8, y=12, collidable=False, block_height=1,
                  group_ids=group, group_leader=False, update_order=3)
        actor = CassetteActor(8, 23)
        result = step_cassette(clock(active=False), (a, b), actor)
        self.assertEqual(result.actor, actor)
        self.assertFalse(any(item.collidable for item in result.blocks))
        self.assertFalse(any(event.kind == "wiggle_up" for event in result.events))

    def test_update_order_can_split_around_player_without_double_tick(self):
        c = clock(update_order=3, beat_index=7, beat_timer=f32(1 / 6 - 1 / 60))
        b = block(update_order=1, index=1, activated=False, collidable=False, y=21)
        actor = CassetteActor(40, 10)
        whole = step_cassette(c, (b,), actor)
        before = step_cassette(c, (b,), actor, before_order=2)
        after = step_cassette(before.clock, before.blocks, before.actor, after_order=2)
        self.assertEqual((whole.clock, whole.blocks, whole.actor), (after.clock, after.blocks, after.actor))
        self.assertTrue(after.blocks[0].activated)
        self.assertFalse(after.blocks[0].collidable)  # manager ran AFTER block.
        next_frame = step_cassette(after.clock, after.blocks, after.actor)
        self.assertTrue(next_frame.blocks[0].collidable)

    def test_disable_moves_before_removing_collision_and_updates_attachment_events(self):
        actor = CassetteActor(8, 20)
        result = step_cassette(clock(active=False), (block(activated=False),), actor)
        self.assertEqual(result.actor.y, 21)
        self.assertFalse(result.blocks[0].collidable)
        self.assertEqual([e.kind for e in result.events], ["shift", "disable"])

    def test_move_into_ceiling_uses_native_crouch_escape_not_instant_death(self):
        # Activation shifts the solid up one pixel under a standing rider.
        b = block(y=21, collidable=False, block_height=1)
        result = step_cassette(clock(active=False), (b,), CassetteActor(8, 21),
                               solid_rects=((-20, -20, 40, 10),))
        self.assertFalse(result.actor.dead)
        self.assertTrue(result.actor.ducking)
        self.assertEqual(result.actor.y, 20)
        self.assertIn("squish_escape", [event.kind for event in result.events])

    def test_missing_group_and_duplicate_order_are_rejected(self):
        with self.assertRaises(ValueError):
            step_cassette(clock(), (block(group_ids=("a", "missing")),), CassetteActor(40, 0))
        with self.assertRaises(ValueError):
            step_cassette(clock(), (block(update_order=0),), CassetteActor(40, 0))

    def test_will_toggle_uses_tracker_order_independent_of_update_order(self):
        c=clock(beat_index=6,beat_timer=f32(1/6-1/60))
        first=block("a",index=1,activated=False,collidable=False,y=22,
                    update_order=2,tracker_order=1)
        second=block("b",index=1,activated=False,collidable=False,y=22,x=32,
                     update_order=3,tracker_order=0)
        result=step_cassette(c,(first,second),CassetteActor(80,0))
        self.assertEqual([event.identity for event in result.events if event.kind=="shift"],["b","a"])

    def test_blocked_activation_still_updates_depth_and_side_counter(self):
        b=block(y=21,activated=True,collidable=False,depth=0,block_height=1)
        result=step_cassette(clock(active=False),(b,),CassetteActor(8,23),
                            solid_rects=((-8,5,40,12),))
        self.assertFalse(result.blocks[0].collidable)
        self.assertEqual(result.blocks[0].depth,8990)
        counters=dict(result.clock.depth_counters)
        self.assertGreater(counters[8990],0)
        self.assertEqual(counters[8990],counters[8995])


if __name__ == "__main__":
    unittest.main()
