"""Coarse guidance cannot bypass the existing full physical replay contract."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import plan_to_route, route_strategy_specs
from celeste_search.assist_routes import start_route_progress, advance_route_progress, route_complete
from celeste_search.maps import MapData, Room
from celeste_search.physics import DASH, RIGHT, Simulator
from celeste_search.search_progress import progress_reporter


class CorridorIntegrationTests(unittest.TestCase):
    def setUp(self):
        room=Room("a",0,0,160,80,("0"*20,)*8+("1"*20,)*2,(),((16.,64.),))
        self.data=MapData("corridor-contract",{"a":room},Path("corridor-contract.bin"))
        self.sim=Simulator(self.data)
        self.start=self.sim.initial_state("a")
        self.route=dict(waypoints=[dict(room="a",x=48,y=56,tile_size=8),
                                   dict(room="a",x=96,y=56,tile_size=8)],
                        segments=[dict(no_dash=False,no_ground=False) for _ in range(2)])
        self.context=patch("celeste_search.assist_planning._context",return_value=(self.data,self.sim))
        self.context.start();self.addCleanup(self.context.stop)

    def walk(self,route=None):
        route=self.route if route is None else route
        state=self.start;phase=start_route_progress(route,state);actions=[]
        for _ in range(100):
            old=state;state=self.sim.step(old,RIGHT);actions.append(RIGHT)
            phase=advance_route_progress(route,phase,old,state,RIGHT)
            if route_complete(route,phase):
                break
        self.assertTrue(route_complete(route,phase))
        return actions,state,phase

    def invoke(self,result,route=None):
        with patch("celeste_search.corridor_planning.search_corridor_routes",return_value=result) as solver:
            value=plan_to_route("corridor-contract.bin",self.start.to_dict(),route or self.route,
                               strategy="ordered_corridor",seconds=1,max_attempts=1)
        self.assertEqual(solver.call_args.args[1],self.start)
        self.assertEqual(solver.call_args.args[2],route or self.route)
        return value

    def test_corridor_recipe_accepts_hard_constraints_and_multigoal_routes(self):
        route=deepcopy(self.route);route["waypoints"]=route["waypoints"][:1]
        route["segments"]=[dict(no_dash=True,no_ground=False)]
        specs=route_strategy_specs(route,self.start,self.data)
        self.assertEqual(specs[0]["name"],"ordered_corridor")
        self.assertEqual(specs[0]["budget_scale"],2.5)
        self.assertTrue(all(spec["family"]=="ordered" for spec in specs))
        multiple=route_strategy_specs(self.route,self.start,self.data)
        self.assertTrue(any(spec["name"]=="ordered_corridor" for spec in multiple))
        self.assertTrue(any(spec["name"]=="ordered_fine" for spec in multiple))

    def test_full_candidate_keeps_public_points_and_only_then_marks_guidance_verified(self):
        actions,state,phase=self.walk()
        topology=dict(active_id="r1",corridors=[dict(id="r1",status="trying",reason="",
            points=[dict(room="a",x=16,y=64),dict(room="a",x=96,y=64)])])
        details=dict(topology=topology,selected_corridor_id="r1")
        packets=[]
        with progress_reporter(packets.append,self.start,self.route):
            result=self.invoke((actions,state,True,1,len(actions),phase,details))
        self.assertTrue(result["can_execute"])
        self.assertEqual(result["route"],self.route)
        self.assertEqual(len(result["waypoint_indices"]),2)
        self.assertEqual(result["waypoint_indices"][-1],len(actions))
        self.assertEqual(packets[-1]["topology"]["corridors"][0]["status"],"verified")
        self.assertEqual(topology["corridors"][0]["status"],"trying")

    def test_solver_claiming_complete_with_a_prefix_cannot_authorize_execution(self):
        actions,state,phase=self.walk()
        details=dict(selected_corridor_id="r1",topology=dict(active_id="r1",corridors=[
            dict(id="r1",status="trying",reason="",points=[dict(room="a",x=16,y=64),
                                                          dict(room="a",x=96,y=64)])]))
        packets=[]
        with progress_reporter(packets.append,self.start,self.route):
            result=self.invoke((actions[:5],state,True,1,len(actions),phase,details))
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["actions"],[])
        self.assertTrue(result["attempts"][0]["replay_rejected"])
        self.assertFalse(any(row["status"]=="verified" for packet in packets
                             for row in packet.get("topology",{}).get("corridors",[])))
        self.assertEqual(packets[-1]["topology"]["corridors"][0]["status"],"failed")

    def test_corridor_solver_cannot_relax_segment_dash_prohibition(self):
        route=deepcopy(self.route);route["segments"][0]["no_dash"]=True
        actions,state,phase=self.walk(route)
        actions[0]|=DASH
        result=self.invoke((actions,state,True,1,len(actions),phase,{}),route)
        self.assertFalse(result["can_execute"])
        self.assertTrue(result["attempts"][0]["replay_rejected"])


if __name__=="__main__":
    unittest.main()
