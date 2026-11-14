"""A completed grid may use 220 ms setup time without losing its field."""
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from celeste_search import detour_guidance
from test_detour_guidance import fixture


class DetourSetupBudgetTests(unittest.TestCase):
    def delayed_grid(self, simulator, deadline_seconds, **options):
        start_time = time.monotonic()
        virtual = {"offset": 0.}
        built = []
        real_grid = detour_guidance._ClearanceGrid

        def grid_with_elapsed_setup(*args, **kwargs):
            grid = real_grid(*args, **kwargs)
            self.assertTrue(grid.complete, "This is a completed small grid, not incomplete geometry")
            built.append(grid)
            # Only this module's monotonic namespace is virtualized. No
            # sleeping, global clock mutation or replacement geometry.
            virtual["offset"] = .220
            return grid

        guide = detour_guidance.StaticDetourGuide(simulator,
            deadline=start_time+deadline_seconds, **options)
        clock = SimpleNamespace(monotonic=lambda: start_time+virtual["offset"])
        return guide, clock, grid_with_elapsed_setup, built

    def test_completed_220ms_grid_keeps_default_field_but_exceeds_old_150ms_slice(self):
        for options, expected in (({}, True), ({"build_seconds": 3.}, True),
                                  ({"build_seconds": .15}, False)):
            with self.subTest(options=options):
                sim, upper, lower, goal = fixture()
                guide, clock, build, built = self.delayed_grid(sim, 1., **options)
                with patch.object(detour_guidance, "time", clock), \
                        patch.object(detour_guidance, "_ClearanceGrid", side_effect=build), \
                        patch.object(sim, "step", side_effect=AssertionError("guidance cannot step physics")):
                    self.assertEqual(guide.prepare(upper, goal), expected)
                    self.assertEqual(guide.score(upper, goal), 0.)
                    if expected:
                        self.assertEqual(guide.build_seconds, .75)
                        self.assertLess(guide.score(lower, goal), 0.)
                        self.assertEqual(guide.cache_info()["references"], 1)
                        self.assertEqual(guide.cache_info()["failures"], 0)
                    else:
                        self.assertEqual(guide.score(lower, goal), 0.)
                        self.assertEqual(guide.cache_info()["references"], 0)
                        self.assertEqual(guide.cache_info()["failures"], 1)
                    self.assertEqual(guide.cache_info()["builds"], 1)
                    self.assertEqual(len(built), 1)

    def test_global_100ms_deadline_still_rejects_completed_220ms_grid_without_physics(self):
        sim, upper, lower, goal = fixture()
        guide, clock, build, built = self.delayed_grid(sim, .100)
        with patch.object(detour_guidance, "time", clock), \
                patch.object(detour_guidance, "_ClearanceGrid", side_effect=build), \
                patch.object(sim, "step", side_effect=AssertionError("guidance cannot step physics")):
            self.assertFalse(guide.prepare(upper, goal))
            self.assertEqual(guide.score(upper, goal), 0.)
            self.assertEqual(guide.score(lower, goal), 0.)
            self.assertFalse(guide.prepare(lower, goal))
        info = guide.cache_info()
        self.assertEqual((info["builds"], info["failures"], info["references"], info["queries"]),
                         (1, 1, 0, 0))
        self.assertEqual(len(built), 1)


if __name__ == "__main__":
    unittest.main()
