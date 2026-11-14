"""Useful ordered refill relays reach real pool slots without extra budgets."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.assist import TargetPlanner
from celeste_search.assist_planning import _refill_guidance_route
from celeste_search.assist_strategies import route_strategy_specs
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import Simulator
from test_assist_portfolio import Clock, FakePool, ORDER, SPECS


def fixture(refill=(144.,120.)):
    entities=(Entity("zipMover",2,200,64,16,16,{},((232.,64.),)),)
    if refill is not None:
        entities+=(Entity("refill",1,*refill,16,16),)
    room=Room("a",0,0,320,160,("0"*40,)*18+("1"*40,)*2,
              entities,((16.,144.),))
    data=MapData("ordered-refill-slots",{"a":room},Path("refill-slots.bin"))
    start=replace(Simulator(data).initial_state(),frame=73,stamina=53.,rem_x=.2)
    route=dict(waypoints=[dict(room="a",x=256,y=104,tile_size=8),
                          dict(room="a",x=16,y=136,tile_size=8)],
               segments=[dict(no_dash=False,no_ground=False) for _ in range(2)])
    return data,start,route


class OrderedRefillPortfolioTests(unittest.TestCase):
    def test_actual_guidance_promotes_both_state_forms_without_mutating_inputs(self):
        data,start,route=fixture()
        before=deepcopy(route)
        guidance,_=_refill_guidance_route(data,start,route)
        self.assertGreater(len(guidance["waypoints"]),len(route["waypoints"]))
        for actual in (start,start.to_dict()):
            with self.subTest(dictionary=isinstance(actual,dict)):
                saved=deepcopy(actual)
                specs=route_strategy_specs(route,actual,data)
                semantic=next(row for row in specs if row["name"]=="ordered_semantic")
                self.assertEqual(specs[0],semantic)
                self.assertEqual(semantic["family"],"ordered")
                self.assertEqual(semantic["portfolio_family"],"refill_relay")
                self.assertNotIn("budget_scale",semantic)
                self.assertNotIn("time_budget_scale",semantic)
                self.assertNotIn("max_seconds",semantic)
                self.assertEqual(actual,saved)
        self.assertEqual(route,before)

    def test_five_workers_start_corridor_and_refill_relay_on_every_retry(self):
        data,start,route=fixture()
        original_names={row["name"] for row in route_strategy_specs(route,start,
            replace(data,rooms={"a":replace(data.rooms["a"],entities=data.rooms["a"].entities[:1])}))}
        for attempt in (0,1,7):
            with self.subTest(attempt=attempt):
                pool,clock=FakePool(),Clock()
                planner=TargetPlanner(workers=5,seconds=8,expansions=10000,pool=pool,clock=clock)
                try:
                    planner.set_map(data)
                    planner.set_deadline(220.)
                    self.assertTrue(planner.request(str(data.source),start,route,attempt,[]))
                    jobs=[config for config,_ in pool.calls]
                    self.assertEqual(len(jobs),5)
                    self.assertEqual([job["strategy"] for job in jobs[:2]],
                                     ["ordered_corridor","ordered_semantic"])
                    self.assertEqual({row["name"] for row in planner.specs},original_names)
                    self.assertEqual((jobs[0]["seconds"],jobs[0]["expansions"]),(90,25000))
                    self.assertEqual((jobs[1]["seconds"],jobs[1]["expansions"]),
                                     (min(20,8*(1+attempt*.25)),10000))
                    self.assertTrue(all(job["start_state"]==start.to_dict() for job in jobs))
                    self.assertTrue(all(job["route"]==route for job in jobs))
                    self.assertEqual(planner.deadline,220.)
                finally:
                    planner.close()

    def test_single_constrained_missing_and_off_path_refills_are_not_promoted(self):
        for reason in ("single","no_dash","no_ground","absent","off_path","endpoint"):
            with self.subTest(reason=reason):
                location=None if reason=="absent" else (144.,16.) if reason=="off_path" else \
                    (256.,112.) if reason=="endpoint" else (144.,120.)
                data,start,route=fixture(location)
                if reason=="single":
                    route={key:value[:1] for key,value in route.items()}
                elif reason in {"no_dash","no_ground"}:
                    route["segments"][1][reason]=True
                specs=route_strategy_specs(route,start,data)
                self.assertFalse(any(row.get("portfolio_family")=="refill_relay" for row in specs))
                self.assertIn("ordered_semantic",{row["name"] for row in specs})

    def test_unmarked_families_keep_existing_retry_rotation(self):
        _,start,route=fixture(None)
        for attempt in (0,1,7):
            pool,clock=FakePool(),Clock()
            planner=TargetPlanner(workers=5,seconds=8,pool=pool,clock=clock)
            try:
                with patch("celeste_search.assist.route_strategy_specs",return_value=deepcopy(SPECS)):
                    planner.request("fixture.bin",start,route,attempt,[])
                offset=(attempt*5)%len(ORDER)
                expected=ORDER[offset:]+ORDER[:offset]
                self.assertEqual(planner.stats()["strategies"],expected)
                self.assertEqual([config["strategy"] for config,_ in pool.calls],expected[:5])
            finally:
                planner.close()


if __name__=="__main__":
    unittest.main()
