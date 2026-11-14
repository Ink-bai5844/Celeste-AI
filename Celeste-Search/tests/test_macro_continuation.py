"""Short macro slices preserve late recipes and exact in-flight simulation."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from celeste_search.macro_continuation import MacroContinuation,MACRO_EXPANSION,MACRO_PAUSE
from celeste_search.physics import DASH
from celeste_search.refill_moves import refill_approach_edges
from celeste_search.wall_transfers import static_wall_transfer_edges
from test_refill_moves import relay_fixture
from test_wall_transfers import fixture


class MacroContinuationTests(unittest.TestCase):
    def compare_sliced(self,helper,sim,start,goal,limit):
        expected_stats={}
        expected=helper(sim,start,goal,deadline=float("inf"),max_frames=180,
                        stats=expected_stats,max_expansions=limit,max_edges=3)
        clock=[0.];raw=sim.step;steps=[]
        def timed(state,mask):
            clock[0]+=.001
            steps.append((state,mask))
            return raw(state,mask)
        handle=MacroContinuation();stats={};turns=[]
        with patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]), \
             patch.object(sim,"step",side_effect=timed):
            for _ in range(4000):
                before=len(steps)
                actual=helper(sim,start,goal,deadline=clock[0]+.01,max_frames=180,
                    stats=stats,max_expansions=limit,max_edges=3,continuation=handle)
                turns.append((len(steps)-before,stats.get("expansions",0)))
                if handle.done:
                    break
        self.assertTrue(handle.done)
        self.assertGreater(handle.resumes,10)
        self.assertLessEqual(max(size for size,count in turns if count<limit),32,
                             "Ordinary slices resume within the current recipe")
        self.assertLessEqual(max(size for size,_ in turns),180,
                             "Only the final already-paid recipe may drain to its bounded end")
        self.assertEqual(stats,expected_stats,"Already simulated prefixes must not be repeated")
        self.assertEqual(actual,expected)
        self.assertTrue(actual)
        for actions,state,_ in actual:
            current=start
            for mask in actions:
                current=raw(current,mask)
                self.assertFalse(current.dead)
            self.assertEqual(current,state)
            self.assertTrue(goal.achieved(current))
        return actual

    def test_late_static_transfer_recipes_survive_many_slices_and_mirroring(self):
        for mirror in (False,True):
            with self.subTest(mirror=mirror):
                self.compare_sliced(static_wall_transfer_edges,*fixture(mirror=mirror),96)

    def test_refill_resume_preserves_rank_resources_and_subpixels(self):
        sim,start,goals=relay_fixture()
        self.compare_sliced(refill_approach_edges,sim,start,goals[0],160)

    def test_no_dash_constraint_still_applies_after_resume(self):
        sim,start,goal=fixture();handle=MacroContinuation();stats={}
        class NoDash:
            no_dash=True
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                self_outer.assertFalse(mask&DASH)
                return sim.step(state,mask)
        self_outer=self;wrapped=NoDash()
        rows=static_wall_transfer_edges(wrapped,start,goal,deadline=float("inf"),
            stats=stats,continuation=handle,max_expansions=96)
        self.assertTrue(handle.done)
        self.assertEqual(rows,[])

    def test_caller_can_publish_only_adopted_results_without_changing_physics(self):
        for family in ("transfer","refill"):
            with self.subTest(family=family):
                if family=="transfer":
                    sim,start,goal=fixture();helper=static_wall_transfer_edges
                    target="celeste_search.wall_transfers.remember_segment"
                else:
                    sim,start,goals=relay_fixture();goal=goals[0];helper=refill_approach_edges
                    target="celeste_search.refill_moves.remember_segment"
                expected=helper(sim,start,goal,deadline=float("inf"),stats={},
                                max_expansions=96,max_edges=3,continuation=MacroContinuation())
                with patch(target) as remembered:
                    actual=helper(sim,start,goal,deadline=float("inf"),stats={},
                        max_expansions=96,max_edges=3,continuation=MacroContinuation(),
                        remember_results=False)
                self.assertTrue(actual)
                self.assertEqual(actual,expected)
                remembered.assert_not_called()

    def test_different_full_origin_cannot_reuse_saved_macro(self):
        sim,start,goal=fixture();handle=MacroContinuation();stats={}
        static_wall_transfer_edges(sim,start,goal,deadline=0,stats=stats,continuation=handle)
        with self.assertRaises(ValueError):
            static_wall_transfer_edges(sim,replace(start,rem_x=.25),goal,
                deadline=float("inf"),stats=stats,continuation=handle)
        handle.release()
        self.assertTrue(handle.done)
        self.assertIsNone(handle.work)

    def test_cooperative_pause_releases_the_turn_without_spending_an_expansion(self):
        handle=MacroContinuation();stats={};events=[]
        def work(saved):
            yield MACRO_PAUSE
            events.append("resumed")
            yield MACRO_EXPANSION
            saved.results=["proved"]
        self.assertEqual(handle.advance("edge",work,deadline=float("inf"),stats=stats),[])
        self.assertEqual(stats,{})
        self.assertEqual(events,[])
        self.assertEqual(handle.advance("edge",work,deadline=float("inf"),stats=stats),["proved"])
        self.assertEqual(stats["expansions"],1)
        self.assertTrue(handle.done)

    def test_last_paid_recipe_finishes_without_spending_a_second_expansion(self):
        sim,start,goal=fixture();handle=MacroContinuation();stats={};clock=[0.]
        from celeste_search.physics import RIGHT
        expected=start
        actions=[RIGHT]*32
        for mask in actions:
            expected=sim.step(expected,mask)
        def work(saved):
            yield MACRO_EXPANSION
            current=start
            for index,mask in enumerate(actions):
                if index%16==0:
                    yield None
                clock[0]+=.01
                current=sim.step(current,mask)
            saved.results=[(actions,current)]
            yield MACRO_EXPANSION
            raise AssertionError("The exhausted budget cannot start another recipe")
        with patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]):
            rows=handle.advance("last",work,deadline=.05,stats=stats,max_expansions=1)
        self.assertEqual(rows,[(actions,expected)])
        self.assertEqual(stats["expansions"],1)
        self.assertFalse(handle.recipe_active)


if __name__=="__main__":unittest.main()
