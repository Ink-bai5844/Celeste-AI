"""A nearby ledge keeps its real short jump and resource-restoring landing."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import start_route_progress, advance_route_progress, route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import CorridorPortalGoal, CorridorRoute
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, SimState, Simulator


class NearbyLandingTests(unittest.TestCase):
    def test_short_jump_lands_and_restores_dash_before_public_goal_without_long_transfer(self):
        for mirror,offset in ((False,(0,0)),(True,(-384,-512))):
            with self.subTest(mirror=mirror):
                ox,oy=offset
                x=lambda value:ox+(192-value if mirror else value)
                entities=(Entity("solid",1,64,32,16,96),Entity("jumpThru",2,80,64,88,0))
                entities=tuple(replace(e,x=ox+(192-e.x-e.width if mirror else e.x),y=oy+e.y)
                               for e in entities)
                room=Room("ledge",ox,oy,192,160,("0"*24,)*20,entities,())
                sim=Simulator(MapData("short-landing",{room.name:room},Path("fixture.bin")))
                start=SimState(room.name,x(84),oy+84,state=1,facing=1 if mirror else -1,
                               dashes=0,stamina=110)
                portal=CorridorPortalGoal("surface",room.name,x(92),oy+64,
                    bounds=(x(92)-4,oy+63,x(92)+4,oy+65),role="support")
                route=dict(waypoints=[dict(room=room.name,x=x(152) if mirror else x(144),
                                          y=oy+56,tile_size=8)],
                           segments=[dict(no_dash=True,no_ground=False)])
                original=deepcopy(route)
                goal=_selected_goal(route["waypoints"][0])[0]
                corridor=CorridorRoute("short",room.name,(portal,),
                    ((start.x,start.y),(portal.x,portal.y),(x(148),oy+64)),(),84.)
                with patch("celeste_search.corridor_planning.generate_corridor_routes",return_value=[corridor]), \
                     patch("celeste_search.wall_transfers.static_wall_transfer_edges",
                           side_effect=AssertionError("short landing must not use long transfer first")):
                    result=search_corridor_routes(sim,start,route,[goal],
                        deadline=time.monotonic()+2,max_expansions=500,max_frames=160)
                self.assertTrue(result[2])
                state=start
                phase=start_route_progress(route,state)
                restored=[]
                for index,mask in enumerate(result[0],1):
                    before=state
                    state=sim.step(state,mask)
                    phase=advance_route_progress(route,phase,before,state,mask)
                    self.assertFalse(state.dead)
                    self.assertFalse(mask&DASH)
                    self.assertIsNone(phase.violation)
                    if portal.achieved(state) and state.dashes==1:
                        restored.append(index)
                self.assertTrue(restored)
                self.assertLess(restored[0],len(result[0]))
                self.assertEqual(state,result[1])
                self.assertTrue(route_complete(route,phase))
                self.assertEqual(route,original)


if __name__=="__main__":
    unittest.main()
