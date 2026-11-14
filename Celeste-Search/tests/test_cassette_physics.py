"""Cassette integration: real simulator collisions, phase and capability gates."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.cassette import CassetteBlockState, CassetteClock, advance_clock
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, RIGHT, DT, SimState, Simulator
from celeste_search.planner import Goal, room_is_supported, search_leg, state_key
from celeste_search.assist_planning import _selected_goal, plan_to_route, target_reached
from celeste_search.routes import snapshot
from celeste_search.safe_handoff import neutral_handoff_safe


def fixture(*extra):
    entity=Entity("cassetteBlock",1,32.,40.,16.,8.,{"index":0})
    room=Room("a",0,0,128,80,("0"*16,)*8+("1"*16,)*2,(entity,*extra),((16.,64.),))
    data=MapData("cassette-fixture",{"a":room},Path("cassette-fixture.bin"))
    clock=CassetteClock(0,0.,0,1.,0,2,True,0,True,True,0,
                        depth=0,actual_depth=-.000001,
                        depth_counters=((-10,.000001),(-5,.000001),(0,.000003),
                                        (10,.000001),(15,.000001),(8990,.000001),(8995,.000001)))
    block=CassetteBlockState("a:1",0,1.,32.,40.,16.,8.,True,True,True,2,
                             True,("a:1",),2,0.,0.,0.,depth=-10,
                             actual_depth=-10.,tracker_order=0)
    simulator=Simulator(data)
    start=replace(simulator.initial_state(),cassette_clock=clock,cassette_blocks=(block,),
                  player_update_order=1,player_depth=0,player_actual_depth=-.000002)
    return data,simulator,start


class CassettePhysicsTests(unittest.TestCase):
    def test_target_hit_rejects_pickup_in_the_neutral_handoff(self):
        _,simulator,state=fixture(Entity("cassette",2,80.,40.,0.,0.))
        goal,target=_selected_goal({"room":"a","x":88,"y":40,"tile_size":8})
        checked=replace(goal,handoff_frames=120,handoff_simulator=simulator)
        unsafe=replace(state,x=94.,y=46.,vx=-108.33328,vy=-60.,
                       on_ground=False,dashes=0)
        self.assertTrue(goal.achieved(unsafe))
        self.assertTrue(target_reached(unsafe,target))
        self.assertFalse(checked.achieved(unsafe))
        self.assertTrue(checked.first_hit_rejected(unsafe))
        tail=simulator.trace(unsafe,[0]*120)
        self.assertEqual(tail[-1].reason,"unsupported_cassette_pickup")
        safe=replace(unsafe,x=92.,y=53.,vx=0.,vy=-105.)
        self.assertTrue(checked.achieved(safe))
        self.assertFalse(checked.first_hit_rejected(safe))
        self.assertTrue(neutral_handoff_safe(simulator,safe,120))

    def test_target_approach_hits_original_tile_safely_without_a_hidden_suffix(self):
        _,simulator,state=fixture(Entity("cassette",2,80.,40.,0.,0.))
        start=replace(state,x=59.,y=64.,vx=0.,vy=0.,on_ground=True)
        goal,target=_selected_goal({"room":"a","x":88,"y":40,"tile_size":8})
        goal=replace(goal,handoff_frames=120,handoff_simulator=simulator)
        result=search_leg(simulator,start,goal,deadline=time.monotonic()+1,
                          max_expansions=100,max_frames=120)
        self.assertTrue(result.reached,result.reason)
        trace=simulator.trace(start,result.actions)
        self.assertEqual(trace[-1],result.state)
        self.assertEqual([i for i,state in enumerate(trace) if target_reached(state,target)],
                         [len(result.actions)])
        self.assertTrue(neutral_handoff_safe(simulator,result.state,120))

    def test_search_prunes_an_unsafe_first_hit_before_a_later_safe_hit(self):
        class FirstHitGoal(Goal):
            def achieved(self,state):
                return state.x>=2
            def first_hit_rejected(self,state):
                return state.x==1
        _,simulator,state=fixture()
        start=replace(state,x=0.)
        def step(current,mask):
            return replace(current,x=current.x+1,frame=current.frame+1)
        with patch.object(simulator,"step",side_effect=step) as run, \
             patch("celeste_search.planner.macros",return_value=[((RIGHT,RIGHT),2)]):
            result=search_leg(simulator,start,FirstHitGoal("target","a",2.,64.),
                              deadline=time.monotonic()+1,max_expansions=3,
                              max_frames=10,allow_unsupported=True)
        self.assertFalse(result.reached)
        self.assertEqual(run.call_count,1)

    def test_serialization_and_search_identity_retain_complete_phase(self):
        _,_,state=fixture()
        self.assertEqual(SimState.from_dict(state.to_dict()),state)
        later=replace(state,cassette_clock=replace(state.cassette_clock,beat_timer=.01))
        self.assertNotEqual(state_key(state),state_key(later))
        self.assertNotEqual(snapshot(state),snapshot(later))
        self.assertNotIn("update_order",snapshot(state)["cassette_clock"])

    def test_room_requires_observed_phase_and_rejects_unknown_destination(self):
        data,simulator,state=fixture()
        self.assertFalse(room_is_supported(simulator,"a"))
        self.assertTrue(room_is_supported(simulator,"a",state))
        for invalid in (replace(state,cassette_clock=None),replace(state,cassette_blocks=()),
                        replace(state,cassette_blocks=(replace(state.cassette_blocks[0],tracker_order=-1),))):
            self.assertFalse(room_is_supported(simulator,"a",invalid))
        with patch("celeste_search.assist_planning._context",return_value=(data,simulator)):
            result=plan_to_route(str(data.source),replace(state,cassette_clock=None,cassette_blocks=()).to_dict(),
                                 {"waypoints":[{"room":"a","x":80,"y":56,"tile_size":8}]},seconds=.1)
        self.assertFalse(result["can_execute"])
        self.assertEqual(result["reason"],"unsupported_room_mechanics")

    def test_cassette_rectangles_participate_in_wall_and_ground_queries(self):
        _,simulator,state=fixture()
        standing=replace(state,x=40,y=40)
        self.assertTrue(simulator.on_ground(standing))
        self.assertTrue(simulator.solid_at("a",40,41,state=state))
        inactive=replace(state,cassette_blocks=(replace(state.cassette_blocks[0],collidable=False),))
        self.assertFalse(simulator.solid_at("a",40,41,state=inactive))
        self.assertFalse(simulator.on_ground(replace(inactive,x=40,y=40)))

    def test_manager_before_player_and_after_player_each_advance_exactly_once(self):
        _,simulator,state=fixture()
        for depth in (-.0000005,-.000003):
            anchor=replace(state,cassette_clock=replace(state.cassette_clock,actual_depth=depth))
            expected,_=advance_clock(anchor.cassette_clock,DT)
            actual=simulator.step(anchor,0)
            self.assertEqual(actual.cassette_clock.beat_timer,expected.beat_timer)
            self.assertEqual(actual.frame,anchor.frame+1)
            self.assertFalse(actual.dead)

    def test_dash_freeze_adds_unscaled_music_time_before_regular_update(self):
        _,simulator,state=fixture()
        # Manager runs after Player; tempo=2 applies only to the regular update.
        state=replace(state,cassette_clock=replace(state.cassette_clock,tempo_mult=2.,actual_depth=-.000003))
        frozen,_=advance_clock(state.cassette_clock,.05,music_time=True)
        expected,_=advance_clock(frozen,DT)
        actual=simulator.step(state,DASH|RIGHT)
        self.assertEqual(actual.cassette_clock.beat_timer,expected.beat_timer)
        self.assertEqual(actual.state,2)

    def test_deactivation_moves_rider_then_removes_collision(self):
        _,simulator,state=fixture()
        block=replace(state.cassette_blocks[0],activated=False)
        state=replace(state,x=40.,y=40.,cassette_blocks=(block,),on_ground=True)
        actual=simulator.step(state,0)
        self.assertEqual(actual.y,41.)
        self.assertFalse(actual.cassette_blocks[0].collidable)
        self.assertEqual(actual.cassette_blocks[0].depth,8990)
        self.assertEqual(actual.lift_timer,10)
        self.assertGreater(actual.lift_vy,59)

    def test_pickup_trigger_rejects_contact_but_preserves_adjacent_safe_space(self):
        _,simulator,state=fixture(Entity("cassette",2,80.,24.,0.,0.))
        touching=simulator.step(replace(state,x=91.,y=29.,on_ground=False),0)
        self.assertTrue(touching.dead)
        self.assertEqual(touching.reason,"unsupported_cassette_pickup")
        clear=simulator.step(replace(state,x=92.,y=29.,on_ground=False),0)
        self.assertFalse(clear.dead)

    def test_same_room_route_can_search_and_replay_with_observed_cassette(self):
        _,simulator,state=fixture()
        goal=Goal("target","a",88.,64.)
        result=search_leg(simulator,state,goal,deadline=time.monotonic()+2,max_expansions=100,max_frames=120)
        self.assertTrue(result.reached,result.reason)
        replay=state
        for action in result.actions:
            replay=simulator.step(replay,action)
        self.assertEqual(replay,result.state)
        self.assertFalse(replay.dead)

    def test_cassette_room_transition_is_not_extrapolated(self):
        data,_,state=fixture()
        other=replace(data.rooms["a"],name="b",x=128,spawns=((144.,64.),),entities=())
        simulator=Simulator(replace(data,rooms={**data.rooms,"b":other}))
        state=replace(state,x=124.,vx=90.)
        moved=simulator.step(state,RIGHT)
        self.assertTrue(moved.dead)
        self.assertEqual(moved.reason,"unsupported_cassette_transition")

    def test_interleaved_falling_block_and_attached_spikes_remain_unsupported(self):
        for entity in (Entity("fallingBlock",2,80.,40.,16.,8.,{"behind":True}),
                       Entity("spikesUp",2,32.,40.,16.,0.)):
            _,simulator,state=fixture(entity)
            self.assertFalse(room_is_supported(simulator,"a",state))

    def test_manager_after_block_activation_waits_for_the_next_block_update(self):
        _,simulator,state=fixture()
        block=replace(state.cassette_blocks[0],depth=10,actual_depth=10.)
        clock=replace(state.cassette_clock,current_index=0,beat_index=7,
                      beat_timer=1/6-DT,actual_depth=-.000003)
        state=replace(state,cassette_clock=clock,cassette_blocks=(block,))
        first=simulator.step(state,0)
        self.assertFalse(first.cassette_blocks[0].activated)
        self.assertTrue(first.cassette_blocks[0].collidable)
        second=simulator.step(first,0)
        self.assertFalse(second.cassette_blocks[0].collidable)


if __name__=="__main__":
    unittest.main()
