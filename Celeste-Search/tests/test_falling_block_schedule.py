"""FallingBlock sequence tests from native coroutine semantics, not path recipes."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator,DT,_f32


def fixture(climb_fall=True, floor=None, platforms=()):
    block=Entity('fallingBlock',1,64.,64.,32.,24.,{'climbFall':climb_fall})
    rows=tuple('0'*24 if floor is None or y*8<floor else '1'*24 for y in range(32))
    room=Room('a',0,0,192,256,rows,(block,)+tuple(platforms),())
    return Simulator(MapData('falling-native-sequence',{'a':room},Path('fixture.bin')))


def native_first_move(wait_contact):
    # Independent Coroutine.Update transcription: positive float waits return
    # even on the update which crosses zero; a yielded None resumes next tick.
    tick=0
    def sequence():
        yield _f32(.2)
        timer=_f32(.4)
        while timer>0 and wait_contact(tick):
            yield None
            timer=_f32(timer-DT)
        return
    it=sequence();wait=0.
    for tick in range(100):
        if wait>0:
            wait=_f32(wait-DT)
        else:
            try:value=next(it)
            except StopIteration:return tick
            if value is not None:wait=value
    raise AssertionError('sequence did not finish')


def first_motion(sim, contact):
    state=SimState('a',100.,80.,state=1,facing=-1)
    for tick in range(80):
        state=contact(tick,state)
        state=sim._advance_mechanisms(state)
        row=next(r for r in state.mechanisms if r[0]=='a:1')
        if row[1]==2:return tick,state
    return None,state


class FallingBlockScheduleTests(unittest.TestCase):
    def test_grab_then_side_jump_keeps_waiting_without_climb_state(self):
        sim=fixture()
        tick,state=first_motion(sim,lambda t,s:replace(s,x=100.,y=80.,state=1 if t==0 else 0,
                                                     vy=-105.,facing=-1))
        self.assertEqual(tick,native_first_move(lambda _:True))
        self.assertEqual(tick,37)
        self.assertGreater(state.mechanisms[0][5],0.)

    def test_leaving_initial_shake_or_contact_wait_resumes_at_native_tick(self):
        for leave in (1,5,14,20,36):
            with self.subTest(leave=leave):
                sim=fixture()
                tick,_=first_motion(sim,lambda t,s:replace(s,x=100. if t<leave else 110.,
                    y=80.,state=1 if t==0 else 0,vy=0.,facing=-1))
                self.assertEqual(tick,native_first_move(lambda t:t<leave))

    def test_neighbor_and_ungrabbed_side_do_not_start_idle_block(self):
        for x,state_id in ((101.,1),(100.,0),(110.,0)):
            sim=fixture();state=SimState('a',x,80.,state=state_id,facing=-1)
            for _ in range(45):state=sim._advance_mechanisms(state)
            self.assertEqual(state.mechanisms[0][1],0)
            self.assertEqual(state.mechanisms[0][4],64.)

    def test_climb_fall_false_only_standing_contact_triggers_and_waits(self):
        sim=fixture(False);side=SimState('a',100.,80.,state=1,facing=-1)
        for _ in range(40):side=sim._advance_mechanisms(side)
        self.assertEqual(side.mechanisms[0][1],0)
        top=SimState('a',80.,64.,on_ground=True)
        for i in range(38):top=sim._advance_mechanisms(top)
        self.assertEqual(top.mechanisms[0][1],2)
        self.assertGreater(top.mechanisms[0][5],0.)

    def test_fall_uses_single_speed_and_counter_before_first_pixel(self):
        sim=fixture();state=SimState('a',120.,80.,mechanisms=(('a:1',1,12,64.,64.,0.),))
        first=sim._advance_mechanisms(state);second=sim._advance_mechanisms(first)
        speed=_f32(500*DT);first_delta=_f32(speed*DT)
        self.assertEqual(first.mechanisms[0][5],speed)
        self.assertAlmostEqual(first.mechanisms[0][4],64.+first_delta,places=10)
        self.assertEqual(round(first.mechanisms[0][4]),64)
        self.assertEqual(round(second.mechanisms[0][4]),64)
        third=sim._advance_mechanisms(second)
        self.assertEqual(round(third.mechanisms[0][4]),65)

    def test_tile_landing_keeps_shake_then_becomes_permanent_and_safe(self):
        sim=fixture(floor=160);state=SimState('a',80.,64.,on_ground=True)
        landing=None
        for tick in range(160):
            state=sim._advance_mechanisms(state)
            phase,timer=state.mechanisms[0][1:3]
            if phase==5 and landing is None:
                landing=tick;self.assertEqual(timer,12);self.assertFalse(sim._safe_ground(state))
            if phase==3:
                self.assertEqual(tick,landing+13)
                self.assertEqual(round(state.mechanisms[0][4]),136)
                self.assertEqual(state.y,136.)
                self.assertTrue(sim._safe_ground(state))
                break
        else:self.fail('no permanent landing')

    def test_jumpthru_landing_polls_support_and_restarts_shake_when_removed(self):
        platform=Entity('jumpThru',2,56.,128.,48.,5.)
        sim=fixture(platforms=(platform,));state=SimState('a',120.,80.,
            mechanisms=(('a:1',2,0,64.,103.,160.),))
        for _ in range(40):
            state=sim._advance_mechanisms(state)
            if state.mechanisms[0][1]==6:break
        self.assertEqual(state.mechanisms[0][1:3],(6,6))
        self.assertEqual(round(state.mechanisms[0][4]),104)
        sim._platforms['a']=()  # An environment fixture removes its supporting platform.
        for _ in range(6):state=sim._advance_mechanisms(state)
        self.assertEqual(state.mechanisms[0][1:3],(6,0))
        state=sim._advance_mechanisms(state)
        self.assertEqual(state.mechanisms[0][1:3],(1,0))

    def test_fall_delay_and_removed_block_remain_explicit_phases(self):
        sim=fixture();state=SimState('a',120.,80.,mechanisms=(('a:1',7,2,64.,64.,0.),))
        for _ in range(2):state=sim._advance_mechanisms(state)
        self.assertEqual(state.mechanisms[0][1:3],(7,0))
        state=sim._advance_mechanisms(state)
        self.assertEqual(state.mechanisms[0][1:3],(1,0))
        state=replace(state,mechanisms=(('a:1',4,0,64.,280.,0.),))
        self.assertFalse(sim._mechanism_rects('a',state))
        self.assertEqual(sim._advance_mechanisms(state).mechanisms,state.mechanisms)


if __name__=='__main__':unittest.main()
