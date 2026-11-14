"""Contact guidance uses the reachable body interval without granting an event."""
from dataclasses import replace
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_routes import _contact_distance, relay_candidate_details
from celeste_search.physics import GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import WallContactGoal, WallGripGoal
from celeste_search.zip_grips import ZipContactGoal


def fixture(name="contact", offset=(0, 0), mirror=False, static=False):
    ox, oy = offset
    def px(x):
        return ox + (384-x if mirror else x)
    def left(x, width):
        return px(x+width) if mirror else px(x)
    entities = () if static else (
        Entity("zipMover", 1, left(80, 24), oy+32, 24, 160,
               {}, ((left(80, 24), oy),)),
        Entity("spikesUp", 2, left(80, 24), oy+32, 24, 0),
        Entity("zipMover", 3, left(160, 24), oy+112, 24, 40,
               {}, ((left(160, 24), oy+8),)),
        Entity("spikesUp", 4, left(160, 24), oy+112, 24, 0),
    )
    grid = []
    for ty in range(32):
        row = []
        for tx in range(48):
            original_x = 47-tx if mirror else tx
            solid = ty >= 24 or (static and 10 <= original_x < 13 and 4 <= ty < 24)
            row.append("1" if solid else "0")
        grid.append("".join(row))
    room = Room(name, ox, oy, 384, 256, tuple(grid), entities, ())
    simulator = Simulator(MapData("contact-range", {name: room}, Path("fixture.bin")))
    mechanisms = tuple((f"{name}:{e.id}", 0, 0, e.x, e.y, 0.)
                       for e in entities if e.kind == "zipMover")
    state = SimState(name, px(40), oy+192, on_ground=True, mechanisms=mechanisms)
    public = Goal("target", name, px(352), oy+8)
    kwargs = dict(kind="wall_grip", room=name, x=px(76), y=oy+44,
                  facing=-1 if mirror else 1, id="first-contact",
                  rect=(left(80, 24), oy+32, left(80, 24)+24, oy+192),
                  identity="" if static else f"{name}:1",
                  mechanism_kind="" if static else "zipMover")
    event = WallContactGoal(**kwargs) if static else ZipContactGoal(**kwargs)
    old = WallGripGoal(**kwargs)
    return simulator, state, public, event, old, px


class MechanismContactGuidanceTests(unittest.TestCase):
    def test_high_guide_does_not_remove_low_local_zip_entry_under_mirror_or_translation(self):
        for name, offset, mirror in (("original", (0, 0), False),
                                     ("renamed", (-720, -1200), False),
                                     ("mirror", (-480, -904), True)):
            with self.subTest(room=name):
                sim, state, public, event, old, _ = fixture(name, offset, mirror)
                before = state.to_dict()
                self.assertEqual(_contact_distance(state, event), 36.)
                self.assertGreater(_contact_distance(state, old), 112.)
                with patch.object(sim, "step", side_effect=AssertionError("guidance cannot simulate")):
                    chosen = relay_candidate_details(sim, state, public, limit=1)[0]
                    alternatives = relay_candidate_details(sim, state, public, limit=32)
                self.assertEqual((chosen["goal"].identity, chosen["goal"].facing),
                                 (event.identity, event.facing))
                self.assertTrue(any(row["identity"] == f"{name}:3"
                                    and row["score"] < chosen["score"] for row in alternatives))
                self.assertEqual({row["goal"].facing for row in alternatives
                                  if row["identity"] == event.identity}, {-1, 1})
                self.assertFalse(event.achieved(state))
                self.assertFalse(public.achieved(state))
                self.assertEqual(state.to_dict(), before)

    def test_zip_distance_uses_actual_eleven_pixel_body_contact_range(self):
        _, state, _, event, _, _ = fixture()
        touching = replace(state, x=76, y=202, state=1, facing=1, on_ground=False)
        # At bottom+10 the 11px body still overlaps the final pixel of the zip.
        self.assertEqual(_contact_distance(touching, event), 0.)
        self.assertTrue(event.achieved(touching))
        below = replace(touching, y=203)
        self.assertEqual(_contact_distance(below, event), 1.)
        self.assertFalse(event.achieved(below))
        at_top = replace(touching, y=32)
        self.assertEqual(_contact_distance(at_top, event), 1.)
        self.assertFalse(event.achieved(at_top))
        self.assertTrue(event.achieved(replace(touching, y=33)))

    def test_zip_distance_tracks_moved_bounds_but_ordinary_grip_keeps_guide_height(self):
        _, state, _, event, old, _ = fixture()
        moved = replace(state, mechanisms=(("contact:1", 3, 31, 80., 0., 0.), state.mechanisms[1]))
        # At the actual upper endpoint the nearest legal foot is y=170.
        self.assertAlmostEqual(_contact_distance(moved, event), math.hypot(36, 22))
        self.assertEqual(_contact_distance(moved, old), math.hypot(36, 148))
        self.assertNotEqual(event.bounds(state), event.bounds(moved))

    def test_real_static_wall_first_touch_is_separate_from_the_high_wall_goal(self):
        for name, offset, mirror in (("wall", (0, 0), False),
                                     ("wall_shift", (-720, -1200), False),
                                     ("wall_mirror", (-480, -904), True)):
            with self.subTest(room=name):
                sim, state, public, event, old, px = fixture(name, offset, mirror, static=True)
                toward = LEFT if mirror else RIGHT
                before = replace(state, x=px(75), y=offset[1]+180,
                                 vx=-90 if mirror else 90, facing=event.facing, on_ground=False)
                contact = sim.step(before, GRAB | toward)
                self.assertFalse(event.achieved(before))
                self.assertFalse(contact.dead)
                self.assertEqual(contact.state, 1)
                self.assertEqual(contact.x, event.x)
                self.assertTrue(event.achieved(contact))
                self.assertEqual(_contact_distance(contact, event), 0.)
                self.assertGreater(_contact_distance(contact, old), 100.)
                self.assertFalse(old.achieved(contact))
                self.assertFalse(public.achieved(contact))

    def test_wall_contact_requires_live_facing_climb_and_positive_stamina(self):
        sim, state, _, event, old, _ = fixture(static=True)
        before = replace(state, x=75, y=180, vx=90, on_ground=False)
        contact = sim.step(before, GRAB | RIGHT)
        self.assertTrue(event.achieved(contact))
        for reason, invalid in (
            ("wrong_face", replace(contact, facing=-1)),
            ("normal_air", replace(contact, state=0)),
            ("dead", replace(contact, dead=True)),
            ("no_stamina", replace(contact, stamina=0)),
            ("other_room", replace(contact, room="other")),
            ("above_wall", replace(contact, y=32)),
            ("below_wall", replace(contact, y=203)),
            ("away_from_wall", replace(contact, x=75)),
        ):
            with self.subTest(reason=reason):
                self.assertFalse(event.achieved(invalid))
        self.assertFalse(old.achieved(contact))


if __name__ == "__main__":
    unittest.main()
