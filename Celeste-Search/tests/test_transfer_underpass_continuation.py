"""Time slicing moving-wall transfers must not restart physical recipes."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from celeste_search.macro_continuation import MacroContinuation
from celeste_search.mechanism_moves import mechanism_transfer_edges
from celeste_search.physics import DASH, Simulator
from celeste_search.wall_underpasses import wall_underpass_edges
from celeste_search.zip_grips import zip_grip_edges, zip_side_grip_goals
from test_mechanism_moves import fixture as mover_fixture
from test_wall_underpasses import fixture as underpass_fixture


class TransferUnderpassContinuationTests(unittest.TestCase):
    def compare_slices(self, helper, sim, origin, target, limit, *, max_frames=220):
        expected_stats = {}
        expected_trace = []
        original_step = sim.step
        def record(before, mask):
            after = original_step(before, mask)
            expected_trace.append((before, mask, after))
            return after
        with patch.object(sim, "step", side_effect=record):
            expected = helper(sim, origin, target, deadline=float("inf"),
                stats=expected_stats, max_expansions=limit, max_frames=max_frames,
                max_edges=4, remember_results=False)
        actual_sim = Simulator(sim.map)
        clock = [100.]
        trace, stats = [], {}
        handle = MacroContinuation()
        raw_step = actual_sim.step
        def timed(before, mask):
            after = raw_step(before, mask)
            trace.append((before, mask, after))
            clock[0] += .001
            return after
        with patch.object(actual_sim, "step", side_effect=timed), \
                patch("celeste_search.macro_continuation.time.monotonic", side_effect=lambda:clock[0]):
            for _ in range(10000):
                result = helper(actual_sim, origin, target, deadline=clock[0] + .007,
                    stats=stats, max_expansions=limit, max_frames=max_frames,
                    max_edges=4, continuation=handle, remember_results=False)
                # Another corridor can overwrite these after a suspension.
                actual_sim.last_step_started_dash = True
                if handle.done:
                    break
        self.assertTrue(handle.done, "Each resume must advance the saved recipe")
        self.assertGreater(handle.resumes, 1)
        self.assertEqual(stats, expected_stats)
        self.assertEqual(handle.expansions, stats["expansions"])
        self.assertEqual(trace, expected_trace, "No old prefix may be replayed or lost")
        self.assertEqual(result, expected)
        self.assertIsNone(handle.work)
        self.assertEqual(handle.children, [])
        for actions, final, _ in result:
            fresh = Simulator(sim.map)
            state = origin
            for action in actions:
                state = fresh.step(state, action)
                self.assertFalse(state.dead)
            self.assertEqual(state, final)
            self.assertTrue(target.achieved(state))
        return result

    def test_airborne_catch_and_actual_zip_launch_preserve_full_trace(self):
        for mirror, offset in ((False, (0, 0)), (True, (-640, -800))):
            with self.subTest(mirror=mirror):
                sim, spring, grip, target = mover_fixture(mirror, offset)
                edges = self.compare_slices(mechanism_transfer_edges, sim, spring, grip, 112)
                self.assertTrue(edges)
                caught = edges[0][1]
                ride = list(zip_grip_edges(sim, caught, grip, target, deadline=float("inf")))[-1]
                launch = ride[1]
                self.assertEqual(launch.state, 1)
                self.assertTrue(launch.mechanisms)
                self.assertNotEqual(launch.lift_vx, 0)
                other = next(g for g in zip_side_grip_goals(sim, launch, target)
                    if g.identity == "relay:6" and g.facing == (-1 if mirror else 1))
                transfers = self.compare_slices(mechanism_transfer_edges, sim, launch, other, 192)
                self.assertTrue(transfers)

    def test_static_underpass_and_refined_timing_preserve_full_trace(self):
        for mirror, offset in ((False, (0, 0)), (True, (-640, -800))):
            with self.subTest(mirror=mirror):
                sim, origin, target = underpass_fixture(mirror, offset)
                result = self.compare_slices(wall_underpass_edges, sim, origin, target, 96)
                self.assertTrue(result)
                self.assertTrue(any("refined" in label for _, _, label in result))

    def test_quota_does_not_restart_and_identity_changes_are_rejected(self):
        for helper, fixture in ((mechanism_transfer_edges, mover_fixture),
                                (wall_underpass_edges, underpass_fixture)):
            sim, origin, target, *_ = fixture()
            handle, stats, clock = MacroContinuation(), {}, [100.]
            raw_step = sim.step
            def timed(before, mask):
                clock[0] += .001
                return raw_step(before, mask)
            with patch.object(sim, "step", side_effect=timed), \
                    patch("celeste_search.macro_continuation.time.monotonic", side_effect=lambda:clock[0]):
                for _ in range(500):
                    helper(sim, origin, target, deadline=clock[0] + .007, stats=stats,
                        max_expansions=stats.get("expansions", 0) + 3, continuation=handle)
                    if handle.done:
                        break
            self.assertTrue(handle.done)
            self.assertEqual(stats["expansions"], 3)
            with self.assertRaises(ValueError):
                helper(sim, replace(origin, frame=origin.frame+1), target,
                    deadline=float("inf"), stats=stats, continuation=handle)

    def test_paid_last_recipe_finishes_but_no_second_recipe_starts(self):
        for helper, fixture in ((mechanism_transfer_edges, mover_fixture),
                                (wall_underpass_edges, underpass_fixture)):
            sim, origin, target, *_ = fixture()
            baseline_stats = {}
            expected = helper(sim, origin, target, deadline=float("inf"),
                stats=baseline_stats, max_expansions=1, max_frames=34)
            stats, clock, handle = {}, [100.], MacroContinuation()
            raw_step = sim.step
            def timed(before, mask):
                clock[0] += .01
                return raw_step(before, mask)
            with patch.object(sim, "step", side_effect=timed), \
                    patch("celeste_search.macro_continuation.time.monotonic", side_effect=lambda:clock[0]):
                actual = helper(sim, origin, target, deadline=clock[0] + .001,
                    stats=stats, max_expansions=1, max_frames=34, continuation=handle)
            self.assertEqual(actual, expected)
            self.assertEqual(stats, baseline_stats)
            self.assertEqual(stats["expansions"], 1)
            self.assertLessEqual(stats["simulated_frames"], 34)
            handle.release()

    def test_constraints_still_apply_to_every_resumed_step(self):
        for helper, fixture in ((mechanism_transfer_edges, mover_fixture),
                                (wall_underpass_edges, underpass_fixture)):
            sim, origin, target, *_ = fixture()
            class Wrapped:
                no_dash = True
                def __getattr__(self, name):
                    return getattr(sim, name)
                def step(self, state, mask):
                    self_test.assertFalse(mask & DASH)
                    return replace(sim.step(state, mask), dead=True, reason="route_limit")
            self_test = self
            stats = {}
            result = helper(Wrapped(), origin, target, deadline=float("inf"),
                stats=stats, max_expansions=8, continuation=MacroContinuation())
            self.assertEqual(result, [])
            self.assertEqual(stats, {"expansions":8, "simulated_frames":8})

    def test_preview_suppression_and_cancel_release(self):
        for helper, fixture, module in (
                (mechanism_transfer_edges, mover_fixture, "mechanism_moves"),
                (wall_underpass_edges, underpass_fixture, "wall_underpasses")):
            sim, origin, target, *_ = fixture()
            with patch(f"celeste_search.{module}.remember_segment") as remember:
                for handle in (None, MacroContinuation()):
                    helper(sim, origin, target, deadline=float("inf"), max_expansions=96,
                        continuation=handle, remember_results=False)
                remember.assert_not_called()
            handle, stats, clock = MacroContinuation(), {}, [100.]
            raw_step = sim.step
            def timed(before, mask):
                clock[0] += .001
                return raw_step(before, mask)
            with patch.object(sim, "step", side_effect=timed), \
                    patch("celeste_search.macro_continuation.time.monotonic", side_effect=lambda:clock[0]):
                helper(sim, origin, target, deadline=clock[0] + .007,
                    stats=stats, max_expansions=96, continuation=handle)
            self.assertFalse(handle.done)
            self.assertIsNotNone(handle.work)
            handle.release()
            self.assertTrue(handle.done)
            self.assertIsNone(handle.work)
            self.assertEqual(handle.results, [])


if __name__ == "__main__":
    unittest.main()
