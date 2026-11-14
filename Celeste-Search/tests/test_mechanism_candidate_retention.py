"""A caught wall retains both returning and unused onward mover capabilities."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_routes import _contact_distance, _rank, relay_candidate_details
from celeste_search.physics import GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.zip_grips import _contact


def fixture(name="retention", offset=(0, 0), mirror=False, extra_grounded_entry=False,
            onward_x=216):
    ox, oy = offset
    def px(x):
        return ox + (384-x if mirror else x)
    def left(x, width):
        return px(x+width) if mirror else px(x)
    entities = [
        Entity("zipMover", 1, 136, 112, 24, 32, {}, ((136, 72),)),
        Entity("spikesUp", 2, 136, 112, 24, 0),
        Entity("zipMover", 3, onward_x, 160, 16, 16, {}, ((onward_x+64, 160),)),
        Entity("spikesUp", 4, onward_x, 160, 16, 0),
    ]
    if extra_grounded_entry:
        entities.extend((Entity("zipMover", 5, 168, 80, 8, 16, {}, ((120, 80),)),
                         Entity("spikesUp", 6, 168, 80, 8, 0),
                         Entity("jumpThru", 7, 184, 100, 8, 0)))
    entities = tuple(replace(e, x=left(e.x, e.width), y=oy+e.y,
                            nodes=tuple((left(x, e.width), oy+y) for x, y in e.nodes))
                     for e in entities)
    grid = []
    for ty in range(32):
        row = []
        for tx in range(48):
            x = 47-tx if mirror else tx
            solid = ty >= 24 or (x == 24 and 6 <= ty < 22) or (x == 21 and 4 <= ty < 8)
            row.append("1" if solid else "0")
        grid.append("".join(row))
    room = Room(name, ox, oy, 384, 256, tuple(grid), entities, ())
    simulator = Simulator(MapData("retention", {name: room}, Path("fixture.bin")))
    records = [(f"{name}:1", 3, 10, left(136, 24), oy+72, 0.),
               (f"{name}:3", 0, 0, left(onward_x, 16), oy+160, 0.)]
    if extra_grounded_entry:
        records.append((f"{name}:5", 0, 0, left(168, 8), oy+80, 0.))
    origin = SimState(name, px(187), oy+100, vx=-90 if mirror else 90,
                      facing=-1 if mirror else 1, on_ground=extra_grounded_entry,
                      mechanisms=tuple(records))
    caught = simulator.step(origin, GRAB | (LEFT if mirror else RIGHT))
    target = Goal("target", name, px(336), oy+8)
    return simulator, caught, target


def transfer_fixture(name="transfer", offset=(0, 0), mirror=False):
    """Catch a real outgoing mover; two onward movers offer both side faces."""
    ox, oy = offset
    def px(x):
        return ox + (512-x if mirror else x)
    def left(x, width):
        return px(x+width) if mirror else px(x)
    entries = ((1, 64, 144), (3, 152, 232), (5, 240, 360))
    entities = tuple(entity for identity, x, endpoint in entries for entity in (
        Entity("zipMover", identity, left(x, 16), oy+128, 16, 32, {},
               ((left(endpoint, 16), oy+128),)),
        Entity("spikesUp", identity+1, left(x, 16), oy+128, 16, 0)))
    room = Room(name, ox, oy, 512, 256, ("0"*64,)*32, entities, ())
    simulator = Simulator(MapData("contact-transfer", {name: room}, Path("transfer.bin")))
    start = SimState(name, px(85), oy+144, vx=90 if mirror else -90,
                     facing=1 if mirror else -1,
                     mechanisms=tuple((f"{name}:{identity}", 0, 0, left(x, 16), oy+128, 0.)
                                      for identity, x, _ in entries))
    caught = simulator.step(start, GRAB | (RIGHT if mirror else LEFT))
    return simulator, caught, Goal("target", name, px(448), oy+32)


def rising_transfer_fixture(name="rising", offset=(0, 0), mirror=False,
                            next_x=152, behind_extra=False):
    ox, oy = offset
    def px(x):
        return ox + (512-x if mirror else x)
    def left(x, width):
        return px(x+width) if mirror else px(x)
    entries = [(1, 64, 128, 64, 64), (3, next_x, 160, next_x+80, 160)]
    if behind_extra:
        entries.append((5, 8, 160, 88, 160))
    entities = tuple(entity for identity, x, y, ex, ey in entries for entity in (
        Entity("zipMover", identity, left(x, 16), oy+y, 16, 32, {}, ((left(ex, 16), oy+ey),)),
        Entity("spikesUp", identity+1, left(x, 16), oy+y, 16, 0)))
    room = Room(name, ox, oy, 512, 256, ("0"*64,)*32, entities, ())
    sim = Simulator(MapData("rising-transfer", {name: room}, Path("rising-transfer.bin")))
    start = SimState(name, px(85), oy+144, vx=90 if mirror else -90,
        facing=1 if mirror else -1,
        mechanisms=tuple((f"{name}:{identity}", 0, 0, left(x, 16), oy+y, 0.)
                         for identity, x, y, _, _ in entries))
    state = sim.step(start, GRAB | (RIGHT if mirror else LEFT))
    for _ in range(16):
        state = sim.step(state, GRAB)
    return sim, state, Goal("target", name, px(496), oy+32)


class MechanismCandidateRetentionTests(unittest.TestCase):
    def test_rising_ride_reserves_forward_lower_near_face_despite_opposite_lift_direction(self):
        for name, offset, mirror in (("upward", (0, 0), False),
                                     ("translated-upward", (-776, -1184), False),
                                     ("mirrored-upward", (440, -824), True)):
            with self.subTest(room=name):
                sim, state, target = rising_transfer_fixture(name, offset, mirror)
                self.assertEqual(state.state, 1)
                self.assertEqual(state.mechanisms[0][1], 2)
                self.assertLess(state.lift_vy, -1)
                self.assertEqual(state.lift_vx, 0)
                before = state.to_dict()
                with patch.object(sim, "step", side_effect=AssertionError("ranking must not simulate")):
                    rows = relay_candidate_details(sim, state, target, limit=6)
                selected = rows[0]
                self.assertEqual((selected["identity"], selected["goal"].facing),
                                 (f"{name}:3", -1 if mirror else 1))
                goal = selected["goal"]
                self.assertGreater(goal.y, state.y)
                self.assertLessEqual((goal.x-state.x)*state.lift_vx +
                                     (goal.y-state.y)*state.lift_vy, 0)
                self.assertLessEqual(_contact_distance(state, goal), 128)
                far = next(row for row in rows if row["identity"] == f"{name}:3"
                           and row["goal"].facing != goal.facing)
                self.assertLess(far["score"], selected["score"])
                self.assertEqual(state.to_dict(), before)

    def test_lower_transfer_beyond_128_stays_ranked_without_reserving_near_face(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                sim, state, target = rising_transfer_fixture(mirror=mirror, next_x=240)
                ranked = _rank(sim, state, target)
                rows = relay_candidate_details(sim, state, target, limit=6)
                lower = [row for row in rows if row["identity"] == "rising:3"]
                self.assertEqual(len(lower), 2)
                self.assertTrue(all(_contact_distance(state, row["goal"]) > 128 for row in lower))
                self.assertEqual([row["goal"].key for row in rows],
                                 [row["goal"].key for row in ranked])
                # In this scene the ordinary rank favors the far side;
                # deleting the radius guard would incorrectly promote near.
                self.assertEqual(rows[0]["goal"].facing, 1 if mirror else -1)

    def test_closer_lower_mover_behind_public_direction_remains_backtracking_only(self):
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                sim, state, target = rising_transfer_fixture(mirror=mirror, behind_extra=True)
                rows = relay_candidate_details(sim, state, target, limit=6)
                selected = rows[0]
                self.assertEqual((selected["identity"], selected["goal"].facing),
                                 ("rising:3", -1 if mirror else 1))
                behind = [row for row in rows if row["identity"] == "rising:5"]
                self.assertEqual(len(behind), 2)
                nearest = min(behind, key=lambda row: _contact_distance(state, row["goal"]))
                self.assertLess(_contact_distance(state, nearest["goal"]),
                                _contact_distance(state, selected["goal"]))
                self.assertLess((nearest["goal"].x-state.x)*(target.x-state.x), 0)
                self.assertGreater((nearest["goal"].x-state.x)*nearest["goal"].facing, 0)
                self.assertLessEqual(len(rows), 6)

    def test_ride_transfer_prefers_near_contact_face_over_cheaper_far_or_distant_launch(self):
        for name, offset, mirror in (("ride", (0, 0), False),
                                     ("translated-ride", (-776, -1184), False),
                                     ("mirrored-ride", (440, -824), True)):
            with self.subTest(room=name):
                sim, state, target = transfer_fixture(name, offset, mirror)
                self.assertEqual(state.state, 1)
                self.assertEqual(state.mechanisms[0][1:3], (1, 7))
                original = state.to_dict()
                with patch.object(sim, "step", side_effect=AssertionError("ranking must not simulate")):
                    rows = relay_candidate_details(sim, state, target, limit=6)
                toward = -1 if mirror else 1
                near = next(row for row in rows if row["identity"] == f"{name}:3"
                            and row["goal"].facing == toward)
                far = next(row for row in rows if row["identity"] == f"{name}:3"
                           and row["goal"].facing == -toward)
                farther_mover = next(row for row in rows if row["identity"] == f"{name}:5"
                                     and row["goal"].facing == toward)
                # Both alternatives have a better optimistic suffix score.
                # The near physical face must still receive the first probe.
                self.assertLess(far["score"], near["score"])
                self.assertLess(farther_mover["score"], near["score"])
                self.assertLess(_contact_distance(state, near["goal"]),
                                _contact_distance(state, farther_mover["goal"]))
                self.assertEqual(rows[0]["goal"], near["goal"])
                self.assertGreater((near["goal"].x-state.x)*near["goal"].facing, 0)
                self.assertLess((far["goal"].x-state.x)*far["goal"].facing, 0)
                self.assertFalse(_contact(state, near["goal"]))
                self.assertFalse(near["goal"].achieved(state))
                self.assertEqual(state.to_dict(), original)

    def test_transfer_promotion_obeys_every_small_quota_and_retains_far_face(self):
        sim, state, target = transfer_fixture()
        for limit in range(9):
            with self.subTest(limit=limit):
                rows = relay_candidate_details(sim, state, target, limit=limit)
                self.assertLessEqual(len(rows), limit)
                self.assertEqual(len(rows), len({row["goal"].key for row in rows}))
                if limit:
                    self.assertEqual((rows[0]["identity"], rows[0]["goal"].facing),
                                     ("transfer:3", 1))
                if limit >= 2:
                    self.assertIn(("transfer:3", -1),
                                  {(row["identity"], row["goal"].facing) for row in rows})

    def test_visited_near_face_does_not_remove_far_backtracking_choice(self):
        sim, state, target = transfer_fixture()
        first = relay_candidate_details(sim, state, target, limit=6)[0]["goal"]
        for visited in ([first], [first.key]):
            with self.subTest(visited=visited):
                rows = relay_candidate_details(sim, state, target, visited=visited, limit=6)
                self.assertNotIn(first.key, {row["goal"].key for row in rows})
                self.assertIn(("transfer:3", -1),
                              {(row["identity"], row["goal"].facing) for row in rows})
                self.assertLessEqual(len(rows), 6)

    def test_wall_catch_keeps_return_and_forward_idle_movers_under_mirror_and_translation(self):
        for name, offset, mirror in (("original", (0, 0), False),
                                     ("renamed", (-720, -1200), False),
                                     ("mirrored", (-480, -904), True)):
            with self.subTest(room=name):
                sim, state, target = fixture(name, offset, mirror)
                self.assertEqual(state.state, 1)
                self.assertFalse(state.on_ground)
                before = state.to_dict()
                with patch.object(sim, "step", side_effect=AssertionError("retention must not simulate")):
                    selected = relay_candidate_details(sim, state, target, limit=6)
                    all_rows = relay_candidate_details(sim, state, target, limit=32)
                self.assertLessEqual(len(selected), 6)
                self.assertEqual((selected[0]["identity"], selected[0]["goal"].facing),
                                 (f"{name}:1", 1 if mirror else -1))
                self.assertIn(f"{name}:3", {row["identity"] for row in selected})
                onward = next(row for row in selected if row["identity"] == f"{name}:3")
                self.assertEqual(onward["goal"].facing, -1 if mirror else 1)
                # Without its own reservation this lower mover loses to many
                # optimistic high walls/supports, including the return mover.
                self.assertGreaterEqual(sum(row["score"] < onward["score"] for row in all_rows), 6)
                self.assertTrue(any(row["family"] == "support" for row in selected))
                self.assertTrue(all(not row["goal"].achieved(state) for row in selected))
                self.assertEqual(state.to_dict(), before)

    def test_small_limits_preserve_nearest_first_and_never_overflow(self):
        sim, state, target = fixture()
        first = relay_candidate_details(sim, state, target, limit=1)
        self.assertEqual([row["identity"] for row in first], ["retention:1"])
        pair = relay_candidate_details(sim, state, target, limit=2)
        self.assertEqual([row["identity"] for row in pair], ["retention:1", "retention:3"])
        for limit in range(9):
            with self.subTest(limit=limit):
                rows = relay_candidate_details(sim, state, target, limit=limit)
                self.assertLessEqual(len(rows), limit)
                self.assertEqual(len(rows), len({row["goal"].key for row in rows}))
                if limit:
                    self.assertEqual(rows[0]["goal"], first[0]["goal"])
                if limit >= 2:
                    self.assertTrue({"retention:1", "retention:3"}.issubset(
                        {row["identity"] for row in rows}))

    def test_retreat_wall_gets_a_contact_slot_without_replacing_either_mover(self):
        for name, offset, mirror in (("retreat", (0, 0), False),
                                     ("shift", (-720, -1200), False),
                                     ("mirror", (-480, -904), True)):
            with self.subTest(room=name):
                sim, state, target = fixture(name, offset, mirror)
                rows = relay_candidate_details(sim, state, target, limit=3)
                self.assertEqual(len(rows), 3)
                self.assertEqual([row["identity"] for row in rows[:2]],
                                 [f"{name}:1", f"{name}:3"])
                wall = rows[2]["goal"]
                self.assertEqual(rows[2]["family"], "wall")
                self.assertEqual(wall.x-state.x, 8 if mirror else -8)
                self.assertEqual(wall.facing, 1 if mirror else -1)
                self.assertLess((wall.x-state.x)*(target.x-state.x), 0)
                self.assertFalse(wall.achieved(state))
                # Support-family reservation must not evict the third
                # reserved retreat contact when all three slots are used.
                wider = relay_candidate_details(sim, state, target, limit=6)
                self.assertIn(wall.key, {row["goal"].key for row in wider})
                self.assertTrue(any(row["family"] == "support" for row in wider))
                excluded = relay_candidate_details(sim, state, target,
                                                   visited=[wall], limit=6)
                self.assertNotIn(wall.key, {row["goal"].key for row in excluded})
                self.assertTrue({f"{name}:1", f"{name}:3"}.issubset(
                    {row["identity"] for row in excluded}))

    def test_visited_faces_are_excluded_without_losing_other_return_or_onward_choices(self):
        sim, state, target = fixture()
        initial = relay_candidate_details(sim, state, target, limit=6)
        return_goal = initial[0]["goal"]
        onward_goal = next(row["goal"] for row in initial if row["identity"] == "retention:3")
        for visited in ([return_goal.key], [onward_goal], [return_goal, onward_goal.key]):
            with self.subTest(visited=visited):
                rows = relay_candidate_details(sim, state, target, visited=visited, limit=6)
                keys = {row["goal"].key for row in rows}
                self.assertFalse(keys.intersection({getattr(value, "key", value) for value in visited}))
                self.assertTrue({"retention:1", "retention:3"}.issubset(
                    {row["identity"] for row in rows}))
                self.assertLessEqual(len(rows), 6)
        all_rows = relay_candidate_details(sim, state, target, limit=32)
        both_onward_faces = [row["goal"] for row in all_rows if row["identity"] == "retention:3"]
        rows = relay_candidate_details(sim, state, target, visited=both_onward_faces, limit=6)
        self.assertNotIn("retention:3", {row["identity"] for row in rows})
        self.assertEqual(rows[0]["identity"], "retention:1")

    def test_three_reserved_capabilities_fit_limit_without_support_replacement_crash(self):
        sim, state, target = fixture(extra_grounded_entry=True)
        self.assertEqual(state.state, 1)
        self.assertTrue(state.on_ground)  # Real catch while standing on a tiny jump-through.
        rows = relay_candidate_details(sim, state, target, limit=3)
        self.assertEqual([row["identity"] for row in rows],
                         ["retention:5", "retention:1", "retention:3"])
        self.assertEqual(len({row["goal"].key for row in rows}), 3)
        all_rows = relay_candidate_details(sim, state, target, limit=32)
        self.assertTrue(any(row["family"] == "support" for row in all_rows))

    def test_distant_onward_mover_does_not_steal_local_return_slot(self):
        sim, state, target = fixture(onward_x=304)
        rows = relay_candidate_details(sim, state, target, limit=2)
        self.assertEqual(rows[0]["identity"], "retention:1")
        self.assertNotIn("retention:3", {row["identity"] for row in rows})


if __name__ == "__main__":
    unittest.main()
