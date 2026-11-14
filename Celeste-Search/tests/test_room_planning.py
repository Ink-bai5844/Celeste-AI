"""One-room control planning: exact offline traces and no game communication."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from celeste_search.maps import Entity, MapData, Room, load_map
from celeste_search.physics import SimState, Simulator
from celeste_search.room_planning import _neighbor_routes, action_fingerprint, plan_room


def fixture(entities=()):
    rows = ("0"*16,)*8+("1"*16,)*2
    first = Room("a", 0, 0, 128, 80, rows, tuple(entities), ((16.0,64.0),))
    second = Room("b", 128, 0, 128, 80, rows, (), ((144.0,64.0),))
    return MapData("fixture", {"a":first, "b":second}, Path("fixture.bin"))


class RoomPlanningTests(unittest.TestCase):
    def test_collected_berry_is_a_stable_checkpoint_before_exit(self):
        data=fixture([Entity("strawberry",5,64,35,0,0)])
        simulator=Simulator(data);start=simulator.initial_state("a")
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=2,expansions=1000)
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["primary_goal"],"a:5")
        self.assertEqual(result["status"],"partial")
        final=SimState.from_dict(result["final_state"])
        self.assertIn("a:5",final.berries)
        self.assertTrue(final.on_ground)
        self.assertEqual(final.vx,0)
        self.assertTrue(result["safe_endpoint"])
        idle=simulator.roll(final,0,120)
        self.assertFalse(idle.dead)
        self.assertEqual((idle.room,idle.x,idle.y),(final.room,final.x,final.y))

    def test_winged_berry_is_collected_without_dash_before_pickup(self):
        data=fixture([Entity("strawberry",5,64,35,0,0,{"winged":True})])
        simulator=Simulator(data);start=simulator.initial_state("a")
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=3,expansions=1500,variant=1)
        self.assertTrue(result["can_execute"],result)
        state=start
        for action in result["actions"]:
            if "a:5" not in state.following and "a:5" not in state.berries:
                self.assertFalse(action & 32)
            state=simulator.step(state,action)
        self.assertIn("a:5",state.berries)
        self.assertNotIn("a:5",state.fled)

    def test_fled_winged_berry_requests_only_current_room_recovery(self):
        data=fixture([Entity("strawberry",5,64,35,0,0,{"winged":True})])
        start=replace(Simulator(data).initial_state("a"),fled=frozenset({"a:5"}))
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=1)
        self.assertFalse(result["can_execute"])
        self.assertTrue(result["requires_room_respawn"])
        self.assertEqual(result["primary_goal"],"a:5")
        self.assertEqual(result["pending_goals"],["a:5"])

    def test_dash_block_is_broken_before_the_exit_goal(self):
        data=fixture([Entity("dashBlock",7,64,32,8,32)])
        simulator=Simulator(data);start=simulator.initial_state("a")
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=3,expansions=1800)
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["primary_goal"],"break:a:7")
        self.assertIn("a:7",result["final_state"]["broken"])
        self.assertTrue(result["goals"][0]["reached"])

    def test_failed_forward_goal_does_not_make_a_cleared_leaf_a_detour(self):
        data=fixture();base=data.rooms["a"]
        rooms={str(i):replace(base,name=str(i),x=i*128,spawns=((i*128+16.,64.),))
               for i in range(4)}
        data=replace(data,rooms=rooms);simulator=Simulator(data)
        state=simulator.initial_state("1")
        ranked=_neighbor_routes(data,simulator,state,{"0","1"},
                    {"failed_goals":{"room:2:272:64":100}})
        self.assertEqual(ranked[0][1].name,"2")
        self.assertEqual(ranked[0][2],"2")
        self.assertEqual(ranked[-1][0][0],float("inf"))

    def test_candidate_preserves_transition_continuation_and_every_frame(self):
        data=fixture()
        start=Simulator(data).initial_state("a")
        with patch("celeste_search.room_planning.load_map", return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=2,expansions=1000)
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["status"],"transition")
        self.assertEqual(len(result["expected"]),len(result["actions"])+1)
        boundary=result["boundary_indices"][0]
        self.assertTrue(all(s["room"]=="a" for s in result["expected"][:boundary]))
        self.assertLess(boundary,len(result["actions"]))
        self.assertTrue(result["allow_room_continuation"])
        self.assertEqual(result["expected"][-1]["room"],"b")
        self.assertEqual(result["fingerprint"],action_fingerprint(result["actions"]))
        self.assertEqual(SimState.from_dict(result["final_state"]).room,"b")

    def test_active_destination_keeps_momentum_through_intermediate_room(self):
        data=fixture();base=data.rooms["a"]
        third=replace(base,name="c",x=256,spawns=((272.,64.),))
        data=replace(data,rooms={**data.rooms,"c":third})
        simulator=Simulator(data);start=simulator.initial_state("a")
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=3,expansions=1500,
                 exploration={"active_destination":"c","navigation_revision":7,
                              "room_visits":{"a":1,"b":1}})
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["destination_room"],"c")
        self.assertEqual(result["navigation_revision"],7)
        self.assertEqual(result["final_state"]["room"],"c")
        self.assertEqual(len(result["boundary_indices"]),2)
        first,second=result["boundary_indices"]
        self.assertTrue(any(result["actions"][first:second]))
        state=start
        for action in result["actions"]:
            state=simulator.step(state,action)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))

    def test_triangle_keeps_target_when_another_frontier_appears(self):
        base=fixture().rooms["a"]
        rooms={name:replace(base,name=name,x=i*128,spawns=((i*128+16.,64.),))
               for i,name in enumerate("abcde")}
        graph={"a":["b","c"],"b":["a","c","d","e"],"c":["a","b","d"],"d":["b","c"],"e":["b"]}
        data=SimpleNamespace(rooms=rooms,neighbors=lambda n:[rooms[k] for k in graph[n]])
        simulator=Simulator(data)
        ledger={"active_destination":"d","recent_edges":["a>b"],"room_visits":{"a":1,"b":1,"c":1}}
        ranked=_neighbor_routes(data,simulator,simulator.initial_state("b"),{"a","b","c"},ledger)
        self.assertEqual(ranked[0][1].name,"d")
        self.assertTrue(all(row[2]=="d" for row in ranked))

    def test_necessary_reverse_edge_remains_available_for_same_destination(self):
        data=fixture();simulator=Simulator(data)
        ranked=_neighbor_routes(data,simulator,simulator.initial_state("b"),{"a","b"},
             {"active_destination":"a","recent_edges":["a>b"],"stalled_edges":{"b>a":3}})
        self.assertEqual(ranked[0][1].name,"a")
        self.assertEqual(ranked[0][2],"a")
        self.assertNotEqual(ranked[0][0][0],float("inf"))

    def test_blacklisted_route_is_not_reissued_as_executable(self):
        data=fixture();start=Simulator(data).initial_state("a")
        with patch("celeste_search.room_planning.load_map", return_value=data):
            first=plan_room("fixture.bin",start.to_dict(),seconds=2,expansions=1000)
            second=plan_room("fixture.bin",start.to_dict(),seconds=2,expansions=1000,
                             forbidden=[first["fingerprint"]])
        self.assertTrue(not second["can_execute"] or second["fingerprint"]!=first["fingerprint"])

    def test_unsupported_current_room_cannot_execute(self):
        data=fixture([Entity("unmodeledFixtureMechanism",1,60,40,16,16)])
        start=Simulator(data).initial_state("a")
        with patch("celeste_search.room_planning.load_map", return_value=data):
            result=plan_room("fixture.bin",start.to_dict(),seconds=1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["status"],"unsupported")
        self.assertEqual(result["actions"],[])

    def test_native_room_one_transitions_to_room_two(self):
        path=Path(__file__).resolve().parents[2]/"GAME"/"Celeste-1"/"Content"/"Maps"/"1-ForsakenCity.bin"
        if not path.exists():
            self.skipTest("owned map not installed")
        data=load_map(path)
        start=Simulator(data).initial_state("1")
        result=plan_room(str(path),start.to_dict(),seconds=8,expansions=4000)
        self.assertTrue(result["can_execute"],result["reason"])
        self.assertEqual(result["status"],"transition")
        self.assertEqual(result["final_state"]["room"],"2")
        boundary=result["boundary_indices"][0]
        self.assertTrue(all(s["room"]=="1" for s in result["expected"][:boundary]))
        self.assertTrue(result["allow_room_continuation"])

    def test_native_room_two_collects_berry_before_exit_or_partial_return(self):
        path=Path(__file__).resolve().parents[2]/"GAME"/"Celeste-1"/"Content"/"Maps"/"1-ForsakenCity.bin"
        if not path.exists():
            self.skipTest("owned map not installed")
        data=load_map(path);simulator=Simulator(data)
        entry=plan_room(str(path),simulator.initial_state("1").to_dict(),seconds=8,expansions=4000)
        self.assertEqual(entry["status"],"transition")
        start=SimState.from_dict(entry["final_state"])
        for _ in range(150):
            start=simulator.step(start,0)
            if start.on_ground and abs(start.vx)<.01:
                break
        result=plan_room(str(path),start.to_dict(),seconds=12,expansions=6000,
                         visited_rooms=["1","2"])
        self.assertTrue(result["can_execute"],result["reason"])
        self.assertIn("2:11",result["final_state"]["berries"])
        self.assertIn(result["status"],{"partial","transition"})
        self.assertTrue(all(s["room"]=="2" for s in result["expected"][:-1]))
        state=start
        for mask in result["actions"]:
            state=simulator.step(state,mask)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))


if __name__=="__main__":
    unittest.main()
