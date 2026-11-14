"""Break actions are discovered without selecting a through-block corridor."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import active_dash_block_edges
from celeste_search.mechanism_planning import _Node, _trim_frontier, search_mechanism_routes
from celeste_search.mechanism_routes import dash_break_goals, _inventory, relay_candidate_details
from celeste_search.physics import Simulator, SimState, LEFT, RIGHT
from celeste_search.planner import SearchResult
from celeste_search.assist_routes import RouteProgress


def fixture(*, mirror=False, offset=(0,0), two=False, can_dash=True):
    ox,oy=offset
    px=lambda x: ox+(320-x if mirror else x)
    blocks=[Entity("dashBlock",7,px(104) if mirror else px(88),oy+128,16,48,
                   attrs={"canDash":can_dash})]
    if two:
        blocks.append(Entity("dashBlock",8,px(200) if mirror else px(184),oy+128,16,48))
    room=Room("active",ox,oy,320,192,("0"*40,)*22+("1"*40,)*2,tuple(blocks),((px(16),oy+176.),))
    sim=Simulator(MapData("active-break",{"active":room},Path("active-break.bin")))
    start=replace(sim.initial_state(),frame=231,rem_x=.125,rem_y=-.0625)
    route=normalize_route(sim.map,dict(room="active",x=px(272),y=oy+168))
    goal=_selected_goal(route["waypoints"][0])[0]
    return sim,start,route,goal


class ActiveDashBreakingTests(unittest.TestCase):
    def test_inventory_proposes_distinct_unbroken_blocks_without_simulation(self):
        sim,start,_,goal=fixture(two=True)
        before=start.to_dict()
        with patch.object(sim,"step",side_effect=AssertionError("discovery cannot simulate or relocate")):
            inventory=[node for node in _inventory(sim,start,goal) if node.family=="dash_break"]
            selected=relay_candidate_details(sim,start,goal,limit=6)
        self.assertEqual({node.identity for node in inventory},{"active:7","active:8"})
        self.assertLessEqual(len(inventory),4)
        self.assertTrue(any(row["family"]=="dash_break" for row in selected))
        self.assertEqual(start.to_dict(),before)
        broken=replace(start,broken=frozenset({"active:7"}))
        self.assertEqual({goal.identity for goal in dash_break_goals(sim,broken)},{"active:8"})
        for value in (False,"false",0):
            blocked,state,_,_=fixture(can_dash=value)
            self.assertEqual(dash_break_goals(blocked,state),[])

    def test_far_approach_and_side_interval_keep_real_prefix_under_mirror_translation(self):
        for mirror,offset in ((False,(0,0)),(True,(-704,-1200))):
            with self.subTest(mirror=mirror):
                sim,start,_,_=fixture(mirror=mirror,offset=offset)
                goal=next(goal for goal in dash_break_goals(sim,start)
                          if goal.dash_dx==(-1 if mirror else 1))
                # The guide is above the floor. A grounded side approach must
                # still be accepted and physically dash through this face.
                goal=replace(goal,y=goal.y-12)
                rows=active_dash_block_edges(sim,start,goal,deadline=time.monotonic()+1.,
                    max_frames=220,max_expansions=320,max_edges=2)
                self.assertTrue(rows)
                for actions,final,_ in rows:
                    state=start
                    for mask in actions:
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                    self.assertEqual(state,final)
                    self.assertIn("active:7",state.broken)
                    self.assertGreater(len(actions),10,"A far approach must include its real walk prefix")
                self.assertFalse(start.broken)

    def solve(self, *, two=True, no_dash=False, fake_break=False, public_blocked=False):
        sim,start,route,goal=fixture(two=two)
        route["segments"][0]["no_dash"]=no_dash
        required={"active:7","active:8"} if two else {"active:7"}
        original_move=active_dash_block_edges
        calls=[]
        def breaking(wrapped,origin,target,**kwargs):
            calls.append((target.identity,origin))
            if fake_break:
                return [([0],replace(origin,broken=origin.broken|{target.identity}),"claimed_break")]
            return original_move(wrapped,origin,target,**kwargs)
        def public_suffix(wrapped,origin,target,**kwargs):
            # Other physics-search policies are disabled in this integration
            # fixture. Only explicit active break actions can open the route.
            if public_blocked or target.kind!="target" or not required.issubset(origin.broken):
                return SearchResult([],origin,False,1,0,"frontier_exhausted")
            state=origin;actions=[]
            for _ in range(min(200,kwargs["max_frames"])):
                if target.achieved(state) or state.dead:
                    break
                state=wrapped.step(state,RIGHT)
                actions.append(RIGHT)
            return SearchResult(actions,state,target.achieved(state),1,len(actions),"frontier_exhausted")
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_moves.active_dash_block_edges",side_effect=breaking))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg",side_effect=public_suffix))
            for path in ("mechanism_moves.mechanism_transfer_edges", "wall_transfers.static_wall_transfer_edges",
                         "wall_underpasses.wall_underpass_edges", "wall_descents.wall_descent_edges",
                         "wall_moves.wall_approach_edges"):
                stack.enter_context(patch("celeste_search."+path,return_value=[]))
            result=search_mechanism_routes(sim,start,route,[goal],deadline=time.monotonic()+2.,
                max_expansions=1200,max_frames=600,use_detours=False)
        return sim,start,route,result,calls

    def test_explicit_two_block_actions_then_original_public_goal_replay(self):
        sim,start,route,result,calls=self.solve()
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertEqual(result[1].broken,frozenset({"active:7","active:8"}))
        self.assertTrue(any(identity=="active:8" and "active:7" in state.broken for identity,state in calls))
        state=start
        for mask in result[0]:
            state=sim.step(state,mask)
            self.assertFalse(state.dead)
        self.assertEqual(state,result[1])
        self.assertGreaterEqual(sum("active_dash_block" in stage for stage in result[6]["stages"]),2)

    def test_no_dash_fake_break_and_partial_completion_never_bypass_validation(self):
        for options in (dict(no_dash=True),dict(fake_break=True),dict(public_blocked=True)):
            with self.subTest(options=options):
                _,_,route,result,calls=self.solve(two=False,**options)
                self.assertFalse(result[2])
                self.assertFalse(route_complete(route,result[5]))
                if options.get("no_dash"):
                    self.assertEqual(calls,[])
                if options.get("fake_break"):
                    self.assertFalse(result[1].broken)

    def test_beam_retains_different_opened_worlds_at_same_contact(self):
        _,start,_,_=fixture()
        nodes=[_Node(start,[],RouteProgress()),
               _Node(replace(start,rem_x=.25),[0],RouteProgress()),
               _Node(replace(start,broken=frozenset({"active:7"})),[0],RouteProgress())]
        for node in nodes:
            node.contact_key="same-contact"
        kept=_trim_frontier([(0.,0),(1.,1),(2.,2)],nodes,width=2)
        self.assertEqual({index for _,index in kept},{0,2})

    def test_already_computed_public_hit_survives_the_last_time_slice(self):
        sim,start,route,_=fixture()
        route["waypoints"][0]["x"]=72
        public=_selected_goal(route["waypoints"][0])[0]
        clock=[10.]
        original=active_dash_block_edges
        calls=[]
        def finish_slice(wrapped,origin,target,**kwargs):
            rows=original(wrapped,origin,target,**kwargs)
            if rows:
                calls.append(rows)
                clock[0]=12.
            return rows
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_planning.time.monotonic",side_effect=lambda:clock[0]))
            stack.enter_context(patch("celeste_search.mechanism_moves.active_dash_block_edges",side_effect=finish_slice))
            for path in ("mechanism_moves.mechanism_transfer_edges", "wall_transfers.static_wall_transfer_edges",
                         "wall_underpasses.wall_underpass_edges", "wall_descents.wall_descent_edges",
                         "wall_moves.wall_approach_edges"):
                stack.enter_context(patch("celeste_search."+path,return_value=[]))
            # The short ordinary public probe is now attempted before private
            # block contacts. Keep it unsuccessful here so the test still
            # exercises a completed macro at the final time boundary.
            with patch("celeste_search.mechanism_planning.search_leg",
                       return_value=SearchResult([],start,False,1,0,"controlled_no_path")) as unrelated_search:
                result=search_mechanism_routes(sim,start,route,[public],deadline=12.,
                    max_expansions=320,max_frames=220,use_detours=False)
                self.assertEqual(unrelated_search.call_count,1,
                    "No additional generic search may start after the completed macro exhausts time")
        self.assertTrue(calls)
        self.assertTrue(result[2],result[6])
        self.assertTrue(route_complete(route,result[5]))
        self.assertFalse(result[1].broken,"The earlier public hit must not invent block destruction")
        state=start
        for action in result[0]:
            state=sim.step(state,action)
        self.assertEqual(state,result[1])


if __name__=="__main__":
    unittest.main()
