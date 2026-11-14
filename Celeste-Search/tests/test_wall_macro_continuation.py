"""Short wall-macro slices retain recipes, native state and nested search work."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.macro_continuation import MacroContinuation
from celeste_search.physics import DASH, Simulator
from celeste_search.planner import SearchNodeBudget, _SEARCH_SESSIONS
from celeste_search.wall_moves import wall_approach_edges, wall_relay_approach_edges
from test_wall_relay_approach import fixture


class WallMacroContinuationTests(unittest.TestCase):
    def sliced(self, function, simulator, origin, goal, *, limit=180, max_edges=2):
        clock=[100.]
        stats={}
        pool=SearchNodeBudget(max_nodes=20000)
        handle=MacroContinuation(pool)
        trace=[]
        raw_step=simulator.step
        def timed(before,mask):
            after=raw_step(before,mask)
            trace.append((before,mask,after))
            clock[0]+=.001
            return after
        calls=0
        with patch.object(simulator,"step",side_effect=timed), \
                patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]):
            while not handle.done and calls<2000:
                function(simulator,origin,goal,deadline=clock[0]+.009,
                    max_frames=180,max_edges=max_edges,stats=stats,
                    max_expansions=limit,continuation=handle)
                # Other suspended routes may overwrite these incidental values
                # and evict the opportunistic global search cache.
                simulator.last_step_started_dash=True
                _SEARCH_SESSIONS.clear()
                calls+=1
        self.assertTrue(handle.done,"Suspension must advance rather than restart")
        self.assertGreater(handle.resumes,1)
        self.assertLessEqual(stats["expansions"],limit)
        self.assertEqual(handle.expansions,stats["expansions"])
        self.assertEqual(pool.nodes,0,"Completed nested searches must release the shared node budget")
        self.assertEqual(handle.children,[])
        return handle.results,stats,trace,calls

    def test_approach_resumes_mid_roll_and_matches_uninterrupted_full_trace(self):
        sim,origin,_,goal=fixture()
        origin=replace(origin,frame=123,stamina=53.,rem_x=.12,rem_y=-.17)
        trace=[];raw=sim.step;stats={}
        def recording(before,mask):
            after=raw(before,mask);trace.append((before,mask,after));return after
        with patch.object(sim,"step",side_effect=recording):
            expected=wall_approach_edges(sim,origin,goal,deadline=time.monotonic()+2,
                max_frames=180,max_edges=2,stats=stats,max_expansions=32)
        other,_,_,_=fixture()
        actual,new_stats,new_trace,_=self.sliced(wall_approach_edges,other,origin,goal,limit=32)
        self.assertTrue(actual)
        self.assertEqual(actual,expected)
        self.assertEqual(new_stats,stats)
        self.assertEqual(new_trace,trace,"No simulated prefix may be replayed or skipped while resuming")
        self.assertTrue(any("runup_jump" in row[2] for row in actual))

    def test_relay_retains_child_search_across_slices_and_lru_eviction(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,880))):
            with self.subTest(mirror=mirror):
                sim,origin,goal,_=fixture(mirror=mirror,offset=offset)
                origin=replace(origin,frame=123,stamina=77.,rem_x=.12)
                expected=wall_relay_approach_edges(sim,origin,goal,deadline=time.monotonic()+2,
                    max_frames=180,max_edges=1,max_expansions=180)
                other,_,_,_=fixture(mirror=mirror,offset=offset)
                actual,_,_,_=self.sliced(wall_relay_approach_edges,other,origin,goal,
                                         limit=180,max_edges=1)
                self.assertTrue(actual)
                self.assertEqual(actual,expected)
                raw=Simulator(other.map)
                state=origin
                for mask in actual[0][0]:
                    self.assertFalse(mask&DASH)
                    state=raw.step(state,mask)
                    self.assertFalse(state.dead)
                    self.assertFalse(raw.last_step_started_dash)
                self.assertEqual(state,actual[0][1])
                self.assertTrue(goal.achieved(state))

    def test_approach_does_not_restart_quota_or_accept_a_changed_anchor(self):
        sim,origin,_,goal=fixture()
        clock=[100.];stats={};handle=MacroContinuation()
        raw=sim.step
        def timed(before,mask):
            clock[0]+=.001
            return raw(before,mask)
        with patch.object(sim,"step",side_effect=timed), \
                patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]):
            wall_approach_edges(sim,origin,goal,deadline=100.005,stats=stats,
                                max_expansions=6,continuation=handle)
            self.assertEqual(stats["expansions"],1)
            self.assertFalse(handle.done)
            for _ in range(300):
                if handle.done:break
                wall_approach_edges(sim,origin,goal,deadline=clock[0]+.005,stats=stats,
                    max_expansions=stats["expansions"]+6,continuation=handle)
            self.assertTrue(handle.done)
            self.assertEqual(stats["expansions"],6)
            with self.assertRaises(ValueError):
                wall_approach_edges(sim,replace(origin,dashes=0),goal,
                    deadline=clock[0]+1,stats=stats,continuation=handle)

    def test_relay_global_budget_pause_returns_without_spinning_or_new_inputs(self):
        sim,origin,goal,_=fixture();handle=MacroContinuation();stats={"expansions":4}
        with patch.object(sim,"step",side_effect=AssertionError("No input at exhausted budget")):
            before=time.monotonic()
            result=wall_relay_approach_edges(sim,origin,goal,deadline=before+1,
                stats=stats,max_expansions=4,continuation=handle)
            self.assertLess(time.monotonic()-before,.1)
        self.assertEqual(result,[])
        self.assertEqual(stats["expansions"],4)
        self.assertFalse(handle.done,"A budget pause must keep the pending setup")
        handle.release()
        self.assertEqual(handle.children,[])

    def test_release_mid_nested_search_frees_every_owned_node(self):
        sim,origin,goal,_=fixture();pool=SearchNodeBudget(20000);handle=MacroContinuation(pool)
        clock=[100.];stats={};raw=sim.step
        def timed(before,mask):
            clock[0]+=.001
            return raw(before,mask)
        with patch.object(sim,"step",side_effect=timed), \
                patch("celeste_search.macro_continuation.time.monotonic",side_effect=lambda:clock[0]):
            for _ in range(1000):
                wall_relay_approach_edges(sim,origin,goal,deadline=clock[0]+.003,
                    max_frames=180,max_edges=1,stats=stats,max_expansions=180,continuation=handle)
                if pool.nodes or handle.done:break
        self.assertGreater(pool.nodes,0,"The fixture must suspend inside the nested graph search")
        handle.release()
        self.assertEqual(pool.nodes,0)
        self.assertIsNone(handle.work)
        self.assertEqual(handle.children,[])

    def test_resumed_relay_preserves_the_callers_no_ground_constraint(self):
        from celeste_search.assist_routes import RouteProgress
        from celeste_search.corridor_planning import _ConstrainedSimulator
        sim,origin,goal,_=fixture()
        route={"waypoints":[{"room":origin.room,"x":72,"y":80,"tile_size":8}],
               "segments":[{"no_dash":True,"no_ground":True}]}
        wrapped=_ConstrainedSimulator(sim,route,no_dash=True,phase=RouteProgress(0,True))
        handle=MacroContinuation();stats={};seen=[];step=wrapped.step
        def recording(before,mask):
            after=step(before,mask);seen.append(after);return after
        with patch.object(wrapped,"step",side_effect=recording):
            for _ in range(100):
                edges=wall_relay_approach_edges(wrapped,origin,goal,
                    deadline=time.monotonic()+.01,max_frames=180,stats=stats,
                    max_expansions=80,continuation=handle)
                if handle.done:break
        self.assertTrue(handle.done)
        self.assertTrue(seen)
        self.assertTrue(any(state.dead and state.reason.startswith("corridor_") for state in seen))
        for actions,expected,_ in edges:
            current=origin
            for mask in actions:
                current=step(current,mask)
                self.assertFalse(current.dead,"Resuming cannot bypass the caller's segment wrapper")
                self.assertFalse(current.on_ground)
            self.assertEqual(current,expected)
            self.assertTrue(goal.achieved(current))


if __name__=="__main__":
    unittest.main()
