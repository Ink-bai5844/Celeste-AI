"""Planner admission stops exactly at certified terminal boundary events."""
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route
from celeste_search.assist_targets import room_exit_targets
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import RIGHT,SimState,Simulator


def fixture(nearby=False):
    source=Room("a",0,0,64,64,("0"*8,)*7+("1"*8,),(),((16.,56.),))
    entities=(Entity("cassetteBlock",1,80 if nearby else 120,24 if nearby else 8,16,16),
              Entity("cassette",2,144,48,0,0))
    destination=Room("b",64,0,96,64,("0"*12,)*7+("1"*12,),entities,((72.,56.),))
    data=MapData("terminal-plan",{"a":source,"b":destination},Path("terminal-plan.bin"))
    sim=Simulator(data)
    target=next(point for point in room_exit_targets(data,"a")
                if point["edge"]=="right" and point["y"]==24)
    return data,sim,target


class TerminalEntryPlanningTests(unittest.TestCase):
    def test_direct_and_ordered_plans_stop_on_the_exact_terminal_frame(self):
        for strategy in ("direct_greedy","ordered_semantic"):
            with self.subTest(strategy=strategy):
                data,sim,target=fixture();start=SimState("a",60,34,vx=240,frame=123)
                native_step=sim.step
                def guarded_step(state,mask):
                    self.assertEqual(state.room,"a","no input is permitted after terminal entry")
                    return native_step(state,mask)
                with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                     patch("celeste_search.planner.macros",return_value=[(RIGHT,4)]), \
                     patch("celeste_search.assist_planning.macros",return_value=[(RIGHT,4)]), \
                     patch.object(sim,"step",side_effect=guarded_step):
                    result=plan_to_route("terminal-plan.bin",start.to_dict(),{"waypoints":[target]},
                            strategy=strategy,seconds=1,expansions=10,max_attempts=1)
                self.assertTrue(result["can_execute"],result)
                self.assertTrue(result["terminal_entry_only"])
                self.assertEqual(result["actions"],[RIGHT])
                self.assertEqual(result["waypoint_indices"],[1])
                self.assertEqual(result["final_state"]["room"],"b")
                self.assertFalse(sim.capability_report("b")["planning_supported"])

    def test_wrong_lane_never_simulates_another_destination_frame(self):
        for strategy in ("direct_greedy","ordered_semantic"):
            data,sim,target=fixture();start=SimState("a",60,46,vx=240)
            native_step=sim.step
            def guarded_step(state,mask):
                self.assertEqual(state.room,"a")
                return native_step(state,mask)
            with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                 patch("celeste_search.planner.macros",return_value=[(RIGHT,4)]), \
                 patch("celeste_search.assist_planning.macros",return_value=[(RIGHT,4)]), \
                 patch.object(sim,"step",side_effect=guarded_step):
                result=plan_to_route("terminal-plan.bin",start.to_dict(),{"waypoints":[target]},
                        strategy=strategy,seconds=1,expansions=10,max_attempts=1)
            self.assertFalse(result["can_execute"])
            self.assertEqual(result["actions"],[])

    def test_interior_remaining_point_and_nearby_unknown_mechanism_stay_unsupported(self):
        data,sim,target=fixture();start=SimState("a",60,34,vx=240)
        routes=({"waypoints":[target,{"room":"b","x":96,"y":24}]},
                {"waypoints":[{"room":"b","x":96,"y":24}]})
        for route in routes:
            with patch("celeste_search.assist_planning._context",return_value=(data,sim)), \
                 patch.object(sim,"step",side_effect=AssertionError("unsupported search")):
                result=plan_to_route("terminal-plan.bin",start.to_dict(),route,
                        strategy="ordered_semantic",seconds=1,max_attempts=1)
            self.assertEqual(result["reason"],"unsupported_room_mechanics")
            self.assertFalse(result["can_execute"])
        near_data,near_sim,near_target=fixture(True)
        with patch("celeste_search.assist_planning._context",return_value=(near_data,near_sim)):
            result=plan_to_route("terminal-plan.bin",start.to_dict(),{"waypoints":[near_target]},
                    strategy="direct_greedy",seconds=1,max_attempts=1)
        self.assertEqual(result["reason"],"unsupported_room_mechanics")


if __name__=="__main__":
    unittest.main()
