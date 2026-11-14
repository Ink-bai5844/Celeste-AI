"""Optional static-wall/landing chains stay geometric and preserve other relays."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.corridor_routes import (FallingBlockTriggerGoal,
    MechanismClearanceGoal, MechanismPassageGoal, RefillPortalGoal,
    _ClearanceGrid, _current_grid, clear_corridor_cache, generate_corridor_routes)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import WallGripGoal
from test_corridor_refill_chains import relay_fixture


def wall_chain_fixture(*, name="chain", offset=(0,0), mirror=False,
                       solid_launch=False, gate=False):
    ox,oy=offset
    entities=[Entity("solid",1,144,96,16,80),Entity("jumpThru",2,208,88,24,0),
              (Entity("solid",3,8,144,40,40) if solid_launch else
               Entity("jumpThru",3,16,144,32,0))]
    if gate:
        entities.append(Entity("fallingBlock",99,184,0,16,192,dict(climbFall=True)))
    entities=tuple(replace(e,x=ox+(256-e.x-e.width if mirror else e.x),y=oy+e.y)
                   for e in entities)
    room=Room(name,ox,oy,256,192,("0"*32,)*24,entities,())
    sim=Simulator(MapData("wall-chain",{name:room},Path("wall-chain.bin")))
    start=SimState(name,ox+(232 if mirror else 24),oy+144,on_ground=True)
    goal=Goal("target",name,ox+(24 if mirror else 232),oy+32)
    return sim,start,goal


def forced(route):
    return tuple(portal for portal in route.portals if "via:"+portal.key in route.signature)


class CorridorWallChainTests(unittest.TestCase):
    def setUp(self):
        clear_corridor_cache()

    def assert_geometric(self,sim,routes,grid=None):
        self.assertTrue(routes)
        self.assertLessEqual(len(routes),6)
        if grid is None:
            grid=_ClearanceGrid(sim,sim.map.rooms[routes[0].room],4,time.monotonic()+2)
        for route in routes:
            record=route.to_dict()
            self.assertEqual(record["status"],"pending")
            self.assertEqual(record["coordinates"],"world_feet")
            self.assertFalse(record["physically_verified"])
            self.assertNotIn("actions",record)
            self.assertNotIn("can_execute",record)
            self.assertTrue(all(grid.clear(point) for point in route.points))
            self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
            for portal in route.portals:
                self.assertIn((portal.x,portal.y),route.points)

    def test_two_forced_contacts_survive_mirror_translation_and_small_portal_cap(self):
        inventories=[]
        for name,offset,mirror in (("chain",(0,0),False),("renamed",(-512,-800),False),
                                   ("mirrored",(-512,-800),True)):
            with self.subTest(name=name):
                sim,start,goal=wall_chain_fixture(name=name,offset=offset,mirror=mirror)
                original=(start,goal)
                with patch.object(sim,"step",side_effect=AssertionError("geometry must not run physics")):
                    routes=generate_corridor_routes(sim,start,goal,max_routes=6,max_portals=2)
                pairs=[forced(route) for route in routes
                       if tuple(p.kind for p in forced(route))==("wall_grip","surface")]
                self.assertTrue(pairs)
                self.assertTrue(all(len(route.portals)<=2 for route in routes))
                ox,oy=offset
                inventories.append({tuple((p.kind,256-(p.x-ox) if mirror else p.x-ox,p.y-oy)
                                           for p in pair) for pair in pairs})
                self.assertEqual((start,goal),original)
                self.assert_geometric(sim,routes)
        self.assertTrue(all(value==inventories[0] for value in inventories[1:]))

    def test_forced_detours_skip_launch_support_and_do_not_repeat_face_heights(self):
        sim,start,goal=wall_chain_fixture(solid_launch=True)
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        walls=[p for route in routes for p in forced(route) if isinstance(p,WallGripGoal)]
        self.assertTrue(walls)
        heights={}
        for wall in walls:
            self.assertNotEqual(wall.rect,(8,144,48,184))
            heights.setdefault((wall.rect,wall.facing),set()).add(wall.y)
        self.assertTrue(all(len(values)==1 for values in heights.values()))
        self.assert_geometric(sim,routes)

    def test_one_portal_limit_does_not_claim_a_missing_second_forced_contact(self):
        sim,start,goal=wall_chain_fixture()
        routes=generate_corridor_routes(sim,start,goal,max_routes=6,max_portals=1)
        self.assertTrue(routes)
        for route in routes:
            self.assertLessEqual(len(route.portals),1)
            requested={part.removeprefix("via:") for part in route.signature if part.startswith("via:")}
            self.assertTrue(requested.issubset({p.key for p in route.portals}))
            self.assertLessEqual(len(requested),1)
        self.assert_geometric(sim,routes)

    def test_wall_and_support_require_distinct_real_contact_states(self):
        sim,start,goal=wall_chain_fixture()
        routes=generate_corridor_routes(sim,start,goal,max_routes=6,max_portals=2)
        wall,support=next(forced(route) for route in routes
                          if tuple(p.kind for p in forced(route))==("wall_grip","surface"))
        at_wall=replace(start,x=wall.x,y=wall.y,on_ground=False,facing=wall.facing,dashes=0)
        self.assertFalse(wall.achieved(at_wall))
        self.assertTrue(wall.achieved(replace(at_wall,state=1)))
        self.assertFalse(wall.achieved(replace(at_wall,state=1,facing=-wall.facing)))
        at_support=replace(at_wall,x=support.x,y=support.y,state=0,vy=20)
        self.assertFalse(support.achieved(at_support))
        self.assertFalse(support.achieved(replace(at_support,on_ground=True,vy=-1)))
        self.assertTrue(support.achieved(replace(at_support,on_ground=True,vy=0)))
        self.assertEqual(at_support.dashes,0)  # Geometry never grants a refill.
        self.assertFalse(goal.achieved(at_support))

    def test_new_wall_and_support_options_do_not_evict_opposite_refill_chains(self):
        sim,start,goal=relay_fixture()
        room=sim.map.rooms[start.room]
        room=replace(room,entities=room.entities+(
            Entity("solid",99,272,168,16,40),Entity("jumpThru",98,16,144,24,0)))
        sim=Simulator(replace(sim.map,rooms={room.name:room}))
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        relays={tuple(p.identity for p in route.portals if isinstance(p,RefillPortalGoal))
                for route in routes}
        self.assertIn(("relay:10","relay:11","relay:12"),relays)
        self.assertIn(("relay:14","relay:13"),relays)
        self.assert_geometric(sim,routes)

    def test_wall_chain_crossing_falling_gate_keeps_trigger_clearance_and_passage(self):
        sim,start,goal=wall_chain_fixture(gate=True)
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        self.assertTrue(any(tuple(p.kind for p in forced(route))==("wall_grip","surface")
                            for route in routes))
        for route in routes:
            self.assertEqual(route.conditional_on,("chain:99",))
            gates=[p for p in route.portals if getattr(p,"identity","")=="chain:99"]
            self.assertEqual([type(p) for p in gates],
                             [FallingBlockTriggerGoal,MechanismClearanceGoal,MechanismPassageGoal])
            self.assertTrue(all(not p.achieved(start) for p in gates))
            self.assertEqual(route.to_dict()["geometry_status"],"requires_mechanism_change")
        opened=_current_grid(sim,sim.map.rooms[start.room],start,4,time.monotonic()+2,
                             frozenset(("chain:99",)))
        self.assert_geometric(sim,routes,opened)


if __name__=="__main__":
    unittest.main()
