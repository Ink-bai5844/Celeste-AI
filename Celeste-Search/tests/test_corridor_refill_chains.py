"""Resource relays remain geometry proposals with real pickup predicates."""
from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.corridor_routes import (RefillPortalGoal,_ClearanceGrid,
    clear_corridor_cache,generate_corridor_routes,_resource_route_priority,CorridorRoute)
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator
from celeste_search.planner import Goal


def relay_fixture(name="relay",offset=(0,0),mirror=False):
    ox,oy=offset
    def x(value):return 320-value if mirror else value
    entities=[Entity("solid",1,80,112,160,24),
              Entity("refill",10,240,56,0,0),Entity("refill",11,176,40,0,0),
              Entity("refill",12,104,32,0,0),Entity("refill",13,104,168,0,0),
              Entity("refill",14,168,168,0,0)]
    entities=tuple(replace(e,x=x(e.x+e.width) if mirror else e.x,y=e.y) for e in entities)
    room=Room(name,ox,oy,320,208,("0"*40,)*26,
        tuple(replace(e,x=e.x+ox,y=e.y+oy) for e in entities),())
    sim=Simulator(MapData("resource-relay",{name:room},Path("relay.bin")))
    start=SimState(name,x(286)+ox,144+oy,dashes=1)
    goal=Goal("target",name,x(60)+ox,106+oy)
    return sim,start,goal


class RefillChainTests(unittest.TestCase):
    def setUp(self):clear_corridor_cache()

    def assert_clear(self,sim,routes):
        grid=_ClearanceGrid(sim,sim.map.rooms[routes[0].room],4,time.monotonic()+2)
        for route in routes:
            self.assertFalse(route.to_dict()["physically_verified"])
            self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
            for portal in route.portals:
                self.assertIn((portal.x,portal.y),route.points)

    def test_three_airborne_refills_and_opposite_bypass_survive_bounded_inventory(self):
        sim,start,goal=relay_fixture()
        before=(start,goal)
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        chains=[tuple(p.identity for p in route.portals if isinstance(p,RefillPortalGoal)) for route in routes]
        self.assertIn(("relay:10","relay:11","relay:12"),chains)
        self.assertTrue(any("relay:14" in chain and "relay:13" in chain for chain in chains))
        self.assertLessEqual(len(routes),6)
        self.assertTrue(all(len(chain)==len(set(chain)) for chain in chains))
        self.assertEqual((start,goal),before)
        self.assert_clear(sim,routes)

    def test_refill_chain_roles_are_invariant_under_translation_rename_and_mirror(self):
        inventories=[]
        for name,offset,mirror in (("relay",(0,0),False),("negative",(-640,-912),False),
                                   ("mirrored",(0,0),True)):
            sim,start,goal=relay_fixture(name,offset,mirror)
            routes=generate_corridor_routes(sim,start,goal,max_routes=6)
            self.assert_clear(sim,routes)
            inventories.append({tuple(int(p.identity.split(":")[-1]) for p in r.portals
                if isinstance(p,RefillPortalGoal)) for r in routes})
        self.assertTrue(all((10,11,12) in inventory and (14,13) in inventory for inventory in inventories))

    def test_small_portal_cap_never_silently_drops_a_forced_resource(self):
        sim,start,goal=relay_fixture()
        routes=generate_corridor_routes(sim,start,goal,max_routes=6,max_portals=2)
        self.assertTrue(routes)
        for route in routes:
            self.assertLessEqual(len(route.portals),2)
            forced={item.removeprefix("via:") for item in route.signature if item.startswith("via:")}
            self.assertTrue(forced.issubset({p.key for p in route.portals}))
        self.assert_clear(sim,routes)

    def test_chain_geometry_remains_blocked_by_solid_between_resources(self):
        sim,start,goal=relay_fixture()
        room=sim.map.rooms[start.room]
        divider=Entity("solid",99,192,0,16,112)
        room=replace(room,entities=(*room.entities,divider))
        sim=Simulator(replace(sim.map,rooms={room.name:room}))
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        self.assertTrue(routes)
        self.assertFalse(any(tuple(p.identity for p in r.portals if isinstance(p,RefillPortalGoal))==
                             ("relay:10","relay:11","relay:12") for r in routes))
        self.assert_clear(sim,routes)

    def test_expired_deadline_does_not_expand_resource_inventory(self):
        sim,start,goal=relay_fixture()
        self.assertEqual(generate_corridor_routes(sim,start,goal,deadline=time.monotonic()-1),[])

    def test_grounded_ascent_prefers_sustained_upper_relay_over_late_airborne_climb(self):
        sim,start,goal=relay_fixture()
        start=replace(start,on_ground=True)
        routes=generate_corridor_routes(sim,start,goal,max_routes=6)
        chain=tuple(p.identity for p in routes[0].portals if isinstance(p,RefillPortalGoal))
        self.assertEqual(chain,("relay:10","relay:11","relay:12"))
        self.assertTrue(any("relay:14" in tuple(p.identity for p in route.portals
                            if isinstance(p,RefillPortalGoal)) for route in routes))

    def test_return_arrival_prioritizes_ready_lower_chain_and_keeps_spent_upper_alternative(self):
        for name,offset,mirror in (("relay",(0,0),False),("elsewhere",(-640,-912),True)):
            sim,original,first=relay_fixture(name,offset,mirror)
            start=replace(original,x=first.x,y=first.y,on_ground=False,vy=160,
                refill_timers=((f"{name}:10",35),(f"{name}:11",70),(f"{name}:12",110)))
            target=Goal("target",name,original.x,original.y)
            routes=generate_corridor_routes(sim,start,target,max_routes=6)
            first_chain=tuple(int(p.identity.split(":")[-1]) for p in routes[0].portals
                              if isinstance(p,RefillPortalGoal))
            self.assertEqual(first_chain,(13,14))
            self.assertTrue(any(tuple(int(p.identity.split(":")[-1]) for p in route.portals
                if isinstance(p,RefillPortalGoal))==(12,11,10) for route in routes))
            ready=generate_corridor_routes(sim,replace(start,refill_timers=()),target,max_routes=6)
            self.assertEqual({r.id for r in routes},{r.id for r in ready})
            self.assert_clear(sim,routes)

    def test_cooldown_score_allows_resources_that_can_return_before_arrival(self):
        portal=RefillPortalGoal("refill","a",100,64,identity="a:1",
                               pickup_rect=(92,50,108,66),role="refill")
        route=CorridorRoute("relay","a",(portal,),((0,64),(100,64),(160,64)),(),160.)
        state=SimState("a",0,64,refill_timers=(("a:1",150),))
        self.assertEqual(_resource_route_priority(route,state)[0],1)
        returning=replace(state,refill_timers=(("a:1",10),))
        self.assertEqual(_resource_route_priority(route,returning)[0],0)
        self.assertEqual(state.refill_timers,(("a:1",150),))


class RefillPortalTests(unittest.TestCase):
    def setUp(self):
        self.goal=RefillPortalGoal("refill","a",64,69.5,identity="a:1",
                                  pickup_rect=(56,56,72,72),role="refill")

    def test_boundary_pickup_uses_native_hurtbox_and_exact_refill_identity(self):
        # This is outside the old +/-6 centre box, but overlaps the native
        # sixteen-pixel refill collider by one pixel in both directions.
        state=SimState("a",75,82,dashes=1,refill_timers=(("a:1",150),))
        self.assertTrue(self.goal.achieved(state))
        for wrong in (replace(state,x=76),replace(state,y=83),replace(state,dashes=0),
                      replace(state,refill_timers=()),replace(state,refill_timers=(("a:2",150),)),
                      replace(state,refill_timers=(("a:1",0),)),replace(state,room="b"),
                      replace(state,dead=True),replace(state,ducking=True)):
            with self.subTest(wrong=wrong):self.assertFalse(self.goal.achieved(wrong))

    def test_real_simulator_pickup_at_boundary_satisfies_portal_without_recentering(self):
        refill=Entity("refill",1,64,64,0,0)
        room=Room("a",0,0,128,128,("0"*16,)*16,(refill,),())
        sim=Simulator(MapData("pickup",{"a":room},Path("pickup.bin")))
        start=SimState("a",75,82,dashes=0)
        self.assertFalse(self.goal.achieved(start))
        after=sim.step(start,0)
        self.assertEqual((after.x,after.y),(75,82))
        self.assertEqual(after.dashes,1)
        self.assertIn(("a:1",150),after.refill_timers)
        self.assertTrue(self.goal.achieved(after))


if __name__=="__main__":unittest.main()
