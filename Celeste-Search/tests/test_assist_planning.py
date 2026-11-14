"""Target assistance never chooses campaign goals or manufactures a spawn."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest

from celeste_search.assist_planning import plan_to_target,target_reached
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import Simulator,SimState
from celeste_search.routes import snapshot


def fixture(entities=()):
    rows=("0"*16,)*8+("1"*16,)*2
    rooms={name:Room(name,index*128,0,128,80,rows,
                    tuple(entities) if index==0 else (),((index*128+16.,64.),))
           for index,name in enumerate("abc")}
    return MapData("assist-fixture",rooms,Path("assist-fixture.bin"))


class TargetAssistTests(unittest.TestCase):
    def plan(self,data,start,target,**kwargs):
        with patch("celeste_search.room_planning.load_map",return_value=data):
            return plan_to_target("assist-fixture.bin",start.to_dict(),target,
                                  seconds=2,expansions=1500,**kwargs)

    def test_replays_from_actual_nonspawn_position_and_inventory(self):
        data=fixture();simulator=Simulator(data)
        start=replace(simulator.initial_state("a"),x=48,vx=60,frame=123,
                      berries=frozenset({"earlier:5"}))
        target={"room":"a","x":96,"y":56,"tile_size":8}
        with patch.object(Simulator,"initial_state",side_effect=AssertionError("must not reset to spawn")):
            result=self.plan(data,start,target)
        self.assertTrue(result["can_execute"],result)
        self.assertTrue(result["reached"])
        self.assertEqual(result["expected"][0],snapshot(start))
        self.assertEqual(result["anchor_state"]["x"],48)
        state=start
        for action in result["actions"]:
            state=simulator.step(state,action)
            self.assertFalse(state.dead)
        self.assertEqual(state,SimState.from_dict(result["final_state"]))
        self.assertTrue(target_reached(state,target))
        self.assertEqual(len(result["expected"]),len(result["actions"])+1)
        self.assertFalse(result["runtime_verified"])

    def test_adjacent_room_has_continuous_verified_inputs(self):
        data=fixture();simulator=Simulator(data);start=simulator.initial_state("a")
        result=self.plan(data,start,{"room":"b","x":152,"y":56})
        self.assertTrue(result["can_execute"],result)
        self.assertTrue(result["allow_room_continuation"])
        self.assertTrue(result["boundary_indices"])
        self.assertEqual(result["final_state"]["room"],"b")
        self.assertLess(result["boundary_indices"][0],len(result["actions"]))

    def test_reachable_distant_room_preserves_two_continuous_transitions(self):
        data=fixture();start=Simulator(data).initial_state("a")
        result=self.plan(data,start,{"room":"c","x":272,"y":56})
        self.assertTrue(result["can_execute"],result)
        self.assertEqual(result["final_state"]["room"],"c")
        self.assertEqual(len(result["boundary_indices"]),2)
        self.assertTrue(result["allow_room_continuation"])

    def test_already_in_target_requires_no_input(self):
        data=fixture();start=Simulator(data).initial_state("a")
        result=self.plan(data,start,{"room":"a","x":16,"y":56})
        self.assertTrue(result["reached"])
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["reason"],"already_at_target")
        self.assertEqual(result["actions"],[])

    def test_solid_outside_and_disconnected_targets_fail_without_retargeting(self):
        data=fixture();start=Simulator(data).initial_state("a")
        disconnected=replace(data.rooms["c"],x=512,spawns=((528.,64.),))
        data=replace(data,rooms={**data.rooms,"c":disconnected})
        cases=[({"room":"a","x":64,"y":64},"target_is_solid"),
               ({"room":"a","x":-8,"y":56},"target_outside_room"),
               ({"room":"c","x":528,"y":56},"target_room_unreachable")]
        for target,reason in cases:
            with self.subTest(reason=reason):
                result=self.plan(data,start,target)
                self.assertFalse(result["can_execute"])
                self.assertFalse(result["reached"])
                self.assertEqual(result["reason"],reason)
                self.assertEqual(result["target"]["x"],target["x"])
                self.assertEqual(result["actions"],[])

    def test_budget_failure_does_not_publish_a_partial_attempt(self):
        data=fixture();start=Simulator(data).initial_state("a")
        with patch("celeste_search.room_planning.load_map",return_value=data):
            result=plan_to_target("assist-fixture.bin",start.to_dict(),
                        {"room":"a","x":120,"y":56},seconds=1,
                        expansions=1,max_attempts=1)
        self.assertEqual(result["reason"],"no_route_within_budget")
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertEqual(result["final_state"],result["anchor_state"])

    def test_selected_tile_does_not_trigger_autonomous_berry_collection(self):
        data=fixture([Entity("strawberry",5,112,35,0,0)])
        start=Simulator(data).initial_state("a")
        result=self.plan(data,start,{"room":"a","x":48,"y":56})
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["objective"],"user_selected_tile")
        self.assertEqual(result["final_state"]["berries"],[])
        self.assertLess(result["final_state"]["x"],56)
        self.assertFalse(result["requires_room_respawn"])

    def test_failed_route_fingerprint_is_not_reissued_from_completed_cache(self):
        data=fixture();simulator=Simulator(data);start=simulator.initial_state("a")
        target={"room":"a","x":80,"y":56}
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            first=plan_to_target("assist-fixture.bin",start.to_dict(),target,seconds=2,max_attempts=1)
            second=plan_to_target("assist-fixture.bin",start.to_dict(),target,seconds=2,max_attempts=2,
                                  forbidden=[first["fingerprint"]])
        self.assertTrue(first["can_execute"])
        self.assertTrue(not second["can_execute"] or second["fingerprint"]!=first["fingerprint"])

    def test_actual_telemetry_predicate_accounts_for_ducked_collider(self):
        obs={"room":"a","x":20,"y":64,"dead":False,"model_state":{"ducking":True}}
        self.assertTrue(target_reached(obs,{"room":"a","x":16,"y":56}))
        self.assertFalse(target_reached(dict(obs,dead=True),{"room":"a","x":16,"y":56}))

    def test_invalid_target_coordinates_raise_instead_of_silently_snapping(self):
        data=fixture();start=Simulator(data).initial_state("a")
        for x in (float("nan"),float("inf")):
            with self.subTest(x=x),self.assertRaises(ValueError):
                self.plan(data,start,{"room":"a","x":x,"y":56})
        for x in (17,16.5):
            result=self.plan(data,start,{"room":"a","x":x,"y":56})
            self.assertEqual(result["reason"],"invalid_tile_origin")
            self.assertEqual(result["target"]["x"],x)


if __name__=="__main__":
    unittest.main()
