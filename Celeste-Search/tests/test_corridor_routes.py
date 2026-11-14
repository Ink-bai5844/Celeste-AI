"""Geometry proposals preserve body clearance, topology, and original goals."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.corridor_routes import (CorridorPortalGoal,_ClearanceGrid,_clearance_grid,
    _display_points,clear_corridor_cache,corridor_cache_info,generate_corridor_routes,
    FallingBlockTriggerGoal,MechanismClearanceGoal,MechanismPassageGoal,_current_grid)
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator,GRAB,UP
from celeste_search.planner import Goal


def fixture(entities=(), *, tiles=(), name="a", offset=(0,0), size=160):
    ox,oy=offset
    rows=tuple("".join("1" if (x,y) in tiles else "0" for x in range(size//8))
               for y in range(size//8))
    room=Room(name,ox,oy,size,size,rows,
              tuple(replace(e,x=e.x+ox,y=e.y+oy) for e in entities),())
    return Simulator(MapData("corridor-fixture",{name:room},Path("corridor-fixture.bin")))


class CorridorRouteTests(unittest.TestCase):
    def setUp(self):
        clear_corridor_cache()

    def assertClear(self,sim,routes):
        for route in routes:
            grid=_ClearanceGrid(sim,sim.map.rooms[route.room],4,time.monotonic()+1)
            self.assertTrue(all(grid.clear(point) for point in route.points))
            self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
            self.assertFalse(route.to_dict()["physically_verified"])
            self.assertEqual(route.to_dict()["coordinates"],"world_feet")

    def test_solid_obstacle_has_structurally_distinct_upper_and_lower_routes(self):
        tiles={(x,y) for x in range(8,12) for y in range(6,12)}
        sim=fixture(tiles=tiles);routes=generate_corridor_routes(sim,SimState("a",24,72),Goal("target","a",136,72))
        self.assertGreaterEqual(len(routes),2)
        self.assertEqual(len({route.signature for route in routes}),len(routes))
        self.assertTrue(any(min(y for _,y in route.points)<=48 for route in routes))
        self.assertTrue(any(max(y for _,y in route.points)>=108 for route in routes))
        self.assertClear(sim,routes)

    def test_oneway_direction_and_both_downward_bypasses_survive_mirroring(self):
        structures=[]
        for mirror in (False,True):
            sim=fixture((Entity("jumpThru",1,32,64,64,8),),size=128)
            start=SimState("a",52 if mirror else 76,32)
            goal=Goal("target","a",52 if mirror else 76,100)
            routes=generate_corridor_routes(sim,start,goal)
            self.assertGreaterEqual(len(routes),2)
            self.assertTrue(any(min(x for x,_ in route.points)<=28 for route in routes))
            self.assertTrue(any(max(x for x,_ in route.points)>=100 for route in routes))
            self.assertClear(sim,routes)
            structures.append(sorted((tuple(p.kind for p in r.portals),
                tuple((128-x if mirror else x,y) for x,y in r.points)) for r in routes))
        self.assertEqual(*structures)
        grid=_ClearanceGrid(sim,sim.map.rooms["a"],4,time.monotonic()+1)
        self.assertTrue(grid.segment_clear((64,100),(64,32)))
        self.assertFalse(grid.segment_clear((64,32),(64,100)))

    def test_wall_connected_to_ceiling_requires_moving_away_then_returning(self):
        sim=fixture((Entity("solid",1,56,0,16,80),),size=128)
        routes=generate_corridor_routes(sim,SimState("a",24,40),Goal("target","a",104,40))
        self.assertTrue(routes)
        self.assertTrue(all(max(y for _,y in r.points)>=92 for r in routes))
        self.assertClear(sim,routes)

    def test_standing_height_and_hazards_close_an_unsafe_narrow_gap(self):
        for gap,expected in ((16,True),(8,False)):
            sim=fixture((Entity("solid",1,56,0,16,64-gap),Entity("solid",2,56,64,16,64)),size=128)
            routes=generate_corridor_routes(sim,SimState("a",24,60),Goal("target","a",104,60))
            self.assertEqual(bool(routes),expected)
            self.assertClear(sim,routes)
        sim=fixture((Entity("solid",1,56,0,16,48),Entity("solid",2,56,64,16,64),
                     Entity("spikesRight",3,56,48,0,16)),size=128)
        self.assertEqual(generate_corridor_routes(sim,SimState("a",24,60),Goal("target","a",104,60)),[])

    def test_resource_chain_and_mandatory_preview_points_are_preserved(self):
        sim=fixture((Entity("refill",1,112,112,0,0),Entity("jumpThru",2,112,80,16,0)))
        routes=generate_corridor_routes(sim,SimState("a",24,128),Goal("target","a",24,16))
        chain=next(route for route in routes if any(p.kind=="refill" for p in route.portals)
                   and any(p.kind=="surface" for p in route.portals))
        for portal in chain.portals:
            self.assertIn((portal.x,portal.y),chain.points)
        self.assertClear(sim,routes)

    def test_portal_reconnection_never_descends_through_oneway_platform(self):
        sim=fixture((Entity("jumpThru",1,32,64,64,8),),size=128)
        grid=_ClearanceGrid(sim,sim.map.rooms["a"],4,time.monotonic()+1)
        portal=CorridorPortalGoal("corridor","a",64,60,bounds=(60,56,68,64))
        points=_display_points(grid,[(64,100),(64,76),(88,76),(88,48)],(portal,))
        self.assertTrue(points)
        self.assertIn((64,60),points)
        self.assertTrue(all(grid.segment_clear(a,b) for a,b in zip(points,points[1:])))

    def test_translation_negative_world_coordinates_and_room_rename_are_invariant(self):
        tiles={(x,y) for x in range(8,12) for y in range(6,12)}
        answers=[]
        for name,(ox,oy) in (("a",(0,0)),("other",(-240,-408))):
            sim=fixture(tiles=tiles,name=name,offset=(ox,oy))
            routes=generate_corridor_routes(sim,SimState(name,24+ox,72+oy),Goal("target",name,136+ox,72+oy))
            self.assertClear(sim,routes)
            answers.append(sorted((tuple(p.kind for p in r.portals),
                                   tuple((x-ox,y-oy) for x,y in r.points)) for r in routes))
        self.assertEqual(*answers)

    def test_support_requires_real_landing_and_exit_goal_is_not_rewritten(self):
        portal=CorridorPortalGoal("surface","a",40,64,bounds=(36,63,44,65),role="support")
        state=SimState("a",40,64,on_ground=True)
        self.assertTrue(portal.achieved(state))
        for wrong in (replace(state,on_ground=False),replace(state,vy=-90),replace(state,dead=True)):
            self.assertFalse(portal.achieved(wrong))
        sim=fixture();start=SimState("a",24,128)
        goal,_=_selected_goal({"room":"a","x":24,"y":-8,"edge":"top","transition_to":"b",
                               "arrival_bounds":[24,-24,32,0]})
        before=repr(goal);routes=generate_corridor_routes(sim,start,goal)
        self.assertTrue(routes);self.assertEqual(repr(goal),before)
        self.assertTrue(all(p.room=="a" for r in routes for p in r.portals))

    def test_cache_is_bounded_geometry_keyed_and_never_stores_partial_grid(self):
        sim=fixture();room=sim.map.rooms["a"]
        first=_clearance_grid(sim,room,4,time.monotonic()+1)
        second=_clearance_grid(sim,room,4,time.monotonic()+2)
        self.assertIs(first.free,second.free)
        self.assertNotEqual(first.deadline,second.deadline)
        changed=fixture((Entity("solid",1,0,0,40,40),))
        other=_clearance_grid(changed,changed.map.rooms["a"],4,time.monotonic()+1)
        self.assertIsNot(first.free,other.free)
        renamed=fixture(name="renamed")
        _clearance_grid(renamed,renamed.map.rooms["renamed"],4,time.monotonic()+1)
        self.assertEqual(corridor_cache_info()["entries"],3)
        for index in range(8):
            renamed=fixture(name="room"+str(index));_clearance_grid(renamed,renamed.map.rooms["room"+str(index)],4,time.monotonic()+1)
        self.assertEqual(corridor_cache_info()["entries"],6)
        clear_corridor_cache()
        with patch("celeste_search.corridor_routes.time.monotonic",side_effect=[0.,2.]):
            partial=_clearance_grid(sim,room,4,1.)
        self.assertFalse(partial.complete)
        self.assertEqual(corridor_cache_info()["entries"],0)

    def test_expired_budget_and_invalid_options_do_not_create_candidates(self):
        sim=fixture();state=SimState("a",24,128);goal=Goal("target","a",24,16)
        self.assertEqual(generate_corridor_routes(sim,state,goal,deadline=time.monotonic()-1),[])
        self.assertEqual(corridor_cache_info()["entries"],0)
        with self.assertRaises(ValueError):generate_corridor_routes(sim,state,goal,cell_size=3)

    def test_falling_occupancy_retains_clear_detour_and_marks_triggered_passage(self):
        sim=fixture((Entity("fallingBlock",7,72,0,16,96,dict(climbFall=True)),))
        state=SimState("a",40,80);goal=Goal("target","a",120,80)
        routes=generate_corridor_routes(sim,state,goal,max_routes=4)
        current=_current_grid(sim,sim.map.rooms['a'],state,4,time.monotonic()+1)
        conservative=[route for route in routes if not route.conditional_on]
        conditional=[route for route in routes if route.conditional_on]
        self.assertTrue(conservative);self.assertTrue(conditional)
        for route in conservative:
            self.assertTrue(all(current.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
        for route in conditional:
            trigger=next(g for g in route.portals if isinstance(g,FallingBlockTriggerGoal))
            clearance=next(g for g in route.portals if isinstance(g,MechanismClearanceGoal))
            self.assertEqual(trigger.identity,"a:7")
            self.assertLess(route.portals.index(trigger),route.portals.index(clearance))
            self.assertTrue(current.clear((trigger.x,trigger.y)))
            self.assertFalse(clearance.achieved(state))
            self.assertEqual(route.to_dict()['geometry_status'],'requires_mechanism_change')
            self.assertFalse(route.to_dict()['physically_verified'])
        self.assertEqual(corridor_cache_info()['entries'],0)

    def test_real_falling_trigger_requires_contact_and_started_phase(self):
        sim=fixture((Entity("fallingBlock",7,72,0,16,96,dict(climbFall=True)),))
        goal=FallingBlockTriggerGoal('wall_grip','a',68,80,identity='a:7',
            rect=(72,0,88,96),facing=1,mechanism_kind='fallingBlock',grip_offset=80)
        before=SimState('a',68,80,state=1,facing=1,on_ground=False,prev_input=GRAB)
        self.assertFalse(goal.achieved(before))
        after=sim.step(before,GRAB|UP)
        self.assertNotEqual(after.mechanisms[0][1],0)
        self.assertTrue(goal.achieved(after))
        self.assertFalse(goal.achieved(replace(after,x=40)))

    def test_clearance_requires_actual_motion_and_attached_spikes_to_clear(self):
        goal=MechanismClearanceGoal('mechanism_clearance','a',68,80,identity='a:7',
            rect=(72,0,88,96),clearance_box=(64,64,100,80),facing=1,
            attached_hazards=((72,-3,88,0),))
        state=SimState('a',68,80)
        self.assertFalse(goal.achieved(state))
        def pose(phase,y):return replace(state,mechanisms=(('a:7',phase,0,72.,float(y),0.),))
        self.assertFalse(goal.achieved(pose(1,0)))
        self.assertFalse(goal.achieved(pose(2,20)))
        self.assertFalse(goal.achieved(pose(2,80))) # body clears, its top spikes do not
        self.assertTrue(goal.achieved(pose(2,84)))
        self.assertTrue(goal.achieved(pose(4,0)))
        self.assertFalse(goal.achieved(replace(pose(4,0),dead=True)))

    def test_dynamic_grid_tracks_moved_removed_and_permanently_landed_blocks(self):
        block=Entity("fallingBlock",7,72,0,16,160,dict(climbFall=True))
        sim=fixture((block,));state=SimState('a',40,80);goal=Goal('target','a',120,80)
        blocked=_current_grid(sim,sim.map.rooms['a'],state,4,time.monotonic()+1)
        removed=replace(state,mechanisms=(('a:7',4,0,72.,200.,0.),))
        opened=_current_grid(sim,sim.map.rooms['a'],removed,4,time.monotonic()+1)
        self.assertFalse(blocked.segment_clear((40,80),(120,80)))
        self.assertTrue(opened.segment_clear((40,80),(120,80)))
        moved=replace(state,mechanisms=(('a:7',2,0,100.,60.,0.),))
        shifted=_current_grid(sim,sim.map.rooms['a'],moved,4,time.monotonic()+1)
        self.assertTrue(shifted.clear((80,40)));self.assertFalse(shifted.clear((104,80)))
        landed=replace(state,mechanisms=(('a:7',3,0,72.,0.,0.),))
        self.assertEqual(generate_corridor_routes(sim,landed,goal),[])
        self.assertTrue(generate_corridor_routes(sim,removed,goal))
        top_only=fixture((replace(block,attrs=dict(climbFall=False)),))
        self.assertEqual(generate_corridor_routes(top_only,state,goal),[])
        self.assertEqual(corridor_cache_info()['entries'],0)

    def test_attached_spike_geometry_moves_and_disappears_with_its_owner(self):
        block=Entity('fallingBlock',7,72,40,16,24,dict(climbFall=True))
        spike=Entity('spikesUp',8,72,40,16,0,{})
        sim=fixture((block,spike));state=SimState('a',40,80)
        grid=_current_grid(sim,sim.map.rooms['a'],state,4,time.monotonic()+1)
        self.assertEqual(grid.hazards,[(72,37,88,40)])
        moved=replace(state,mechanisms=(('a:7',2,0,72.,80.,30.),))
        shifted=_current_grid(sim,sim.map.rooms['a'],moved,4,time.monotonic()+1)
        self.assertEqual(shifted.hazards,[(72,77,88,80)])
        removed=replace(moved,mechanisms=(('a:7',4,0,72.,200.,0.),))
        gone=_current_grid(sim,sim.map.rooms['a'],removed,4,time.monotonic()+1)
        self.assertEqual(gone.hazards,[])

    def test_conditional_trigger_pairs_survive_small_portal_limit_and_translation(self):
        structures=[]
        for name,(ox,oy) in [('a',(0,0)),('translated',(-192,-256))]:
            sim=fixture((Entity('fallingBlock',7,72,0,16,96,dict(climbFall=True)),),name=name,offset=(ox,oy))
            state=SimState(name,40+ox,80+oy);goal=Goal('target',name,120+ox,80+oy)
            routes=generate_corridor_routes(sim,state,goal,max_routes=4,max_portals=1)
            signatures=[]
            for route in routes:
                triggers=[g for g in route.portals if isinstance(g,FallingBlockTriggerGoal)]
                clearances=[g for g in route.portals if isinstance(g,MechanismClearanceGoal)]
                self.assertEqual(len(triggers),len(clearances))
                signatures.append((bool(route.conditional_on),tuple(g.kind for g in route.portals)))
            self.assertTrue(any(row[0] for row in signatures))
            structures.append(signatures)
        self.assertEqual(*structures)

    def test_single_gate_scenario_bypasses_optional_falling_block_without_trigger(self):
        gate=Entity('fallingBlock',1,112,0,16,160,dict(climbFall=True))
        optional=Entity('fallingBlock',2,64,48,16,48,dict(climbFall=True))
        sim=fixture((gate,optional));state=SimState('a',24,80)
        routes=generate_corridor_routes(sim,state,Goal('target','a',144,80),max_routes=6)
        only_gate=[route for route in routes if route.conditional_on==('a:1',)]
        self.assertTrue(only_gate)
        self.assertEqual(routes[0].conditional_on,('a:1',))
        gate_open=_current_grid(sim,sim.map.rooms['a'],state,4,time.monotonic()+1,frozenset(('a:1',)))
        for route in only_gate:
            self.assertFalse(any(getattr(portal,'identity','')=='a:2' for portal in route.portals))
            self.assertTrue(all(gate_open.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
        self.assertEqual(state.mechanisms,())

    def test_two_required_gates_keep_bounded_combination_fallback(self):
        blocks=(Entity('fallingBlock',1,56,0,16,160,dict(climbFall=True)),
                Entity('fallingBlock',2,112,0,16,160,dict(climbFall=True)))
        sim=fixture(blocks);state=SimState('a',24,80)
        routes=generate_corridor_routes(sim,state,Goal('target','a',144,80),max_routes=3)
        self.assertTrue(routes)
        self.assertTrue(all(set(route.conditional_on)=={'a:1','a:2'} for route in routes))
        for route in routes:
            stages=[(type(portal).__name__,portal.identity) for portal in route.portals
                    if isinstance(portal,(FallingBlockTriggerGoal,MechanismClearanceGoal))]
            self.assertEqual(stages,[('FallingBlockTriggerGoal','a:1'),('MechanismClearanceGoal','a:1'),
                                     ('FallingBlockTriggerGoal','a:2'),('MechanismClearanceGoal','a:2')])

    def test_clearance_is_followed_by_actual_player_passage_with_body_clear_bounds(self):
        block=Entity('fallingBlock',7,72,0,16,96,dict(climbFall=True))
        sim=fixture((block,));start=SimState('a',40,80)
        routes=generate_corridor_routes(sim,start,Goal('target','a',120,80),max_portals=1)
        conditional=[route for route in routes if route.conditional_on]
        self.assertTrue(conditional)
        opened=_current_grid(sim,sim.map.rooms['a'],start,4,time.monotonic()+1,frozenset(('a:7',)))
        for route in conditional:
            index=next(i for i,portal in enumerate(route.portals) if isinstance(portal,MechanismClearanceGoal))
            clearance,passage=route.portals[index:index+2]
            self.assertIsInstance(passage,MechanismPassageGoal)
            self.assertEqual(passage.identity,clearance.identity)
            moved=replace(start,mechanisms=(('a:7',4,0,72.,200.,0.),))
            self.assertTrue(clearance.achieved(moved))
            self.assertFalse(passage.achieved(moved))
            self.assertTrue(passage.achieved(replace(moved,x=passage.x,y=passage.y)))
            left,top,right,bottom=passage.bounds
            self.assertTrue(all(opened.clear((x,y)) for x in range(int(left),int(right)+1)
                                for y in range(int(top),int(bottom)+1)))


if __name__=="__main__":unittest.main()
