"""Zip gate geometry is conditional; only live movement and passage certify it."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import start_route_progress,advance_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator,_mechanism_clearance_edges
from celeste_search.corridor_routes import (ZipMoverTriggerGoal,MechanismClearanceGoal,
    MechanismPassageGoal,ZipMoverPassageGoal,clear_corridor_cache,generate_corridor_routes,_current_grid)
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import Simulator,SimState,GRAB,UP,DASH
from celeste_search.planner import Goal


def fixture(entities,*,width=192,height=192,offset=(0,0),floor=False):
    ox,oy=offset
    es=tuple(replace(e,x=e.x+ox,y=e.y+oy,
        nodes=tuple((x+ox,y+oy) for x,y in e.nodes)) for e in entities)
    tiles=('0'*(width//8),)*(height//8-2)+(('1' if floor else '0')*(width//8),)*2
    r=Room('zip',ox,oy,width,height,tiles,es,((24+ox,80+oy),))
    return Simulator(MapData('zip-gates',{'zip':r},Path('zip-gates.bin')))


def mover(identifier=1,x=80,y=0,w=16,h=192,end=(80,208)):
    return Entity('zipMover',identifier,x,y,w,h,{},(end,))


class ZipConditionalCorridorTests(unittest.TestCase):
    def setUp(self):clear_corridor_cache()

    def test_four_movement_directions_keep_ordered_identity_conditions(self):
        for end in ((80,208),(80,-208),(160,0),(0,0)):
            with self.subTest(end=end):
                clear_corridor_cache();sim=fixture((mover(end=end),));state=SimState('zip',24,80)
                before=state.to_dict()
                page=generate_corridor_routes(sim,state,Goal('target','zip',160,80),
                    max_routes=4,deadline=time.monotonic()+1)
                self.assertTrue(page)
                occupied=_current_grid(sim,sim.map.rooms['zip'],state,4,time.monotonic()+1)
                for route in page:
                    self.assertEqual(route.conditional_on,('zip:1',))
                    portals=[p for p in route.portals if getattr(p,'identity','')=='zip:1']
                    self.assertEqual([type(p) for p in portals],
                        [ZipMoverTriggerGoal,MechanismClearanceGoal,ZipMoverPassageGoal])
                    trigger,clearance,passage=portals
                    self.assertEqual(clearance.mechanism_kind,'zipMover')
                    self.assertTrue(occupied.clear((trigger.x,trigger.y)))
                    self.assertFalse(trigger.achieved(state));self.assertFalse(clearance.achieved(state))
                    data=route.to_dict()
                    self.assertFalse(data['physically_verified'])
                    self.assertEqual(data['conditions'][0]['kind'],'zip_clearance')
                    self.assertFalse(data['conditions'][0]['physically_verified'])
                    self.assertEqual(data['portals'][0]['condition'],'actual_grip_and_triggered')
                self.assertEqual(state.to_dict(),before)

    def test_two_zip_gates_cannot_be_erased_by_opening_only_one(self):
        for offset in ((0,0),(-704,-1200)):
            with self.subTest(offset=offset):
                ox,oy=offset
                sim=fixture((mover(1,x=56,end=(56,208)),mover(2,x=112,end=(112,-208))),offset=offset)
                state=SimState('zip',24+ox,80+oy);goal=Goal('target','zip',168+ox,80+oy)
                page=generate_corridor_routes(sim,state,goal,max_routes=4,max_portals=1,
                    deadline=time.monotonic()+1)
                self.assertTrue(page)
                for route in page:
                    self.assertEqual(route.conditional_on,('zip:1','zip:2'))
                    events=[(type(p).__name__,p.identity) for p in route.portals
                            if isinstance(p,(ZipMoverTriggerGoal,MechanismClearanceGoal,MechanismPassageGoal))]
                    self.assertEqual(events,[(kind,identity) for identity in ('zip:1','zip:2')
                        for kind in ('ZipMoverTriggerGoal','MechanismClearanceGoal','ZipMoverPassageGoal')])

    def test_return_phase_never_counts_as_removal_and_attached_spikes_follow(self):
        goal=MechanismClearanceGoal('mechanism_clearance','zip',76,80,
            identity='zip:1',rect=(80,32,96,96),clearance_box=(80,60,96,80),
            attached_hazards=((80,29,96,32),),mechanism_kind='zipMover')
        start=SimState('zip',76,80)
        pose=lambda phase,y:replace(start,mechanisms=(('zip:1',phase,10,80.,float(y),0.),))
        self.assertFalse(goal.achieved(pose(4,32)))
        self.assertFalse(goal.achieved(pose(4,80)),'Top spikes still occupy the desired clearance')
        self.assertTrue(goal.achieved(pose(4,84)))
        self.assertFalse(goal.achieved(pose(0,84)),'Idle guesses do not certify a triggered gate')
        self.assertTrue(replace(goal,mechanism_kind='fallingBlock').achieved(pose(4,32)))

    def test_real_grip_trigger_wait_and_full_state_replay_preserve_restrictions(self):
        sim=fixture((mover(y=32,h=64,end=(80,112)),Entity('spikesUp',2,80,32,16,0)),floor=True)
        start=SimState('zip',76,96,state=1,facing=1,dashes=1,stamina=110,
            frame=321,rem_y=.125,prev_input=GRAB,mechanisms=(('zip:1',0,0,80.,32.,0.),))
        trigger=ZipMoverTriggerGoal('wall_grip','zip',76,44,identity='zip:1',
            rect=(80,32,96,96),facing=1,mechanism_kind='zipMover',grip_offset=12)
        self.assertFalse(trigger.achieved(start))
        after=sim.step(start,GRAB|UP)
        self.assertTrue(trigger.achieved(after),'Actual low-side contact need not hit the top guide')
        self.assertFalse(trigger.achieved(replace(after,x=40)))
        clearance=MechanismClearanceGoal('mechanism_clearance','zip',76,96,
            identity='zip:1',rect=(80,32,96,96),clearance_box=(80,32,96,56),
            attached_hazards=((80,29,96,32),),mechanism_kind='zipMover')
        route=dict(waypoints=[dict(room='zip',x=152,y=64,tile_size=8)],
                   segments=[dict(no_dash=True,no_ground=True)])
        wrapped=_ConstrainedSimulator(sim,route,no_dash=True)
        stats=dict(expansions=0,simulated_frames=0)
        edges=_mechanism_clearance_edges(wrapped,after,clearance,
            deadline=time.monotonic()+1,max_frames=150,stats=stats,max_expansions=6)
        self.assertTrue(edges);self.assertLessEqual(stats['expansions'],6)
        for actions in edges:
            state=start;phase=start_route_progress(route,start)
            for mask in [GRAB|UP]+actions:
                before=state;state=sim.step(state,mask)
                phase=advance_route_progress(route,phase,before,state,mask)
                self.assertFalse(state.dead);self.assertFalse(phase.violation)
                self.assertFalse(mask&DASH)
            self.assertTrue(clearance.achieved(state))
            self.assertEqual(state.frame,start.frame+1+len(actions))
            self.assertNotEqual(state.mechanisms,start.mechanisms)
        self.assertEqual(start.mechanisms,(('zip:1',0,0,80.,32.,0.),))

    def test_stationary_or_missing_zip_destination_does_not_open_geometry(self):
        for e in (mover(end=(80,0)),replace(mover(),nodes=())):
            sim=fixture((e,));s=SimState('zip',24,80)
            self.assertEqual(generate_corridor_routes(sim,s,Goal('target','zip',160,80),
                deadline=time.monotonic()+1),[])

    def test_budget_exhaustion_cannot_advance_a_conditional_zip(self):
        sim=fixture((mover(),));s=SimState('zip',76,80,state=1,facing=1,
            mechanisms=(('zip:1',1,7,80.,0.,0.),))
        g=MechanismClearanceGoal('mechanism_clearance','zip',76,80,identity='zip:1',
            rect=(80,0,96,192),clearance_box=(80,64,96,80),mechanism_kind='zipMover')
        with patch.object(sim,'step',side_effect=AssertionError('expired edge cannot simulate')):
            for deadline,budget in ((time.monotonic()-1,6),(time.monotonic()+1,0)):
                self.assertEqual(_mechanism_clearance_edges(sim,s,g,deadline=deadline,
                    max_frames=150,stats={'expansions':0,'simulated_frames':0},max_expansions=budget),[])


if __name__=='__main__':unittest.main()
