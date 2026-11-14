"""Short physical approaches to observed beat platforms.

The templates depend only on relative support geometry. Every input, including
button releases and horizontal braking, runs through the unchanged simulator.
Returned states retain the observed clock and all intervening mechanisms.
"""
from __future__ import annotations

import math
import time

from .physics import DASH, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment


def cassette_support_edges(simulator, start, goal, *, deadline=math.inf,
                           max_frames=120, max_edges=6, stats=None):
    """Try launch windows observed in real neutral forecasts, then approaches.

    A late active block can disappear before a long approach finishes. Forecast
    its next real activation while the player remains supported, and prefer a
    launch shortly before that activation. These are ordering hints only: every
    waiting and jumping frame remains in the returned physical edge.
    """
    if start.dead or start.cassette_clock is None or not start.on_ground or start.vy < 0:
        return
    target=next((block for block in start.cassette_blocks
                 if block.id==getattr(goal,'entity_key','')),None)
    if target is None:
        return
    wait_states=[start]
    activations=[]
    deactivation=None
    was_active=target.collidable
    for tick in range(1,min(180,max_frames-2)+1):
        if time.monotonic()>=deadline:
            return
        waiting=simulator.step(wait_states[-1],0)
        if stats is not None:
            stats['simulated_frames']=stats.get('simulated_frames',0)+1
        if waiting.dead or waiting.room!=start.room or not waiting.on_ground:
            break
        wait_states.append(waiting)
        block=next(block for block in waiting.cassette_blocks if block.id==target.id)
        if block.collidable and not was_active:
            activations.append(tick)
        elif was_active and not block.collidable and deactivation is None:
            deactivation=tick
        was_active=block.collidable
    timed=[max(0,tick-lead) for tick in activations for lead in (36,20,0)]
    flight=max(30,min(80,abs(start.x-(target.x+target.width/2))*.65))
    late=target.collidable and deactivation is not None and deactivation<flight+10
    delays=list(dict.fromkeys((timed+[0] if late else [0]+timed)+[16,32,48]))
    emitted = 0
    for delay in delays:
        if stats is not None and stats.get("expansions", 0) >= stats.get("max_expansions", math.inf):
            return
        if delay>=len(wait_states) or delay>=max_frames-2:
            continue
        waiting=wait_states[delay]
        wait_actions=[0]*delay
        for actions, arrival, label, states in _cassette_jump_edges(
                simulator, waiting, goal, deadline=deadline, max_frames=max_frames-delay,
                max_edges=max_edges-emitted, stats=stats):
            combined = wait_actions+actions
            remember_segment(start, arrival, wait_states[:delay+1]+states[1:])
            yield combined, arrival, f"wait:{delay}:"+label
            emitted += 1
            if emitted >= max_edges:
                return


def _cassette_jump_edges(simulator, start, goal, *, deadline=math.inf,
                        max_frames=120, max_edges=6, stats=None):
    """Enumerate live landings with different jump and dash timing.

    An inactive target remains eligible: its achieved predicate requires real
    collision at arrival. No beat, resource or position is assigned here.
    """
    if (start.dead or start.cassette_clock is None or not start.on_ground
            or start.vy < 0 or max_frames < 2):
        return
    target = next((block for block in start.cassette_blocks
                   if block.id == getattr(goal, "entity_key", "")), None)
    if target is None:
        return
    recipes = [(hold, None, False) for hold in (6, 12, 24)]
    if start.dashes > 0:
        recipes += [(hold, dash, up) for hold in (12, 24)
                    for dash in (8, 16, 24) for up in (False, True)]
    seen = set()
    emitted = 0
    for hold, dash_tick, dash_up in recipes:
        for bias in (0., -.35, .35):
            if time.monotonic() >= deadline:
                return
            if stats is not None and stats.get("expansions", 0) >= stats.get("max_expansions", math.inf):
                return
            if stats is not None:
                stats["expansions"] = stats.get("expansions", 0)+1
            state = start
            actions, states = [], [state]
            release = int(bool(start.prev_input & JUMP))
            # Air acceleration is finite: neutral starts braking a few pixels
            # before the requested landing point, with correction if it drifts.
            for tick in range(min(max_frames, 120)):
                if time.monotonic() >= deadline:
                    return
                block = next((block for block in state.cassette_blocks if block.id == target.id), None)
                if block is None:
                    break
                aim_x = block.x + block.width * (.5 + bias)
                distance = aim_x - state.x
                brake = max(2., abs(state.vx) * .055)
                horizontal = (RIGHT if distance > brake else LEFT if distance < -brake else 0)
                local_tick = tick-release
                mask = horizontal
                if local_tick < 0:
                    mask = 0
                elif local_tick < hold:
                    mask |= JUMP
                if local_tick == dash_tick:
                    direction = RIGHT if distance >= 0 else LEFT
                    mask = DASH | direction | (UP if dash_up else 0)
                successor = simulator.step(state, mask)
                if stats is not None:
                    stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
                actions.append(mask)
                states.append(successor)
                state = successor
                if state.dead or state.room != start.room:
                    break
                if tick > release and goal.achieved(state):
                    signature = (round(state.x/3), state.dashes, round(state.vx/30),
                                 state.cassette_clock.current_index,
                                 state.cassette_clock.beat_index % 8,
                                 round(state.cassette_clock.beat_timer*60))
                    if signature not in seen:
                        seen.add(signature)
                        label = f"jump:{hold}:dash:{dash_tick}:up:{int(dash_up)}:aim:{bias}"
                        yield list(actions), state, label, states
                        emitted += 1
                        if emitted >= max_edges:
                            return
                    break
                # A missed landing below the target cannot be repaired by a
                # further jump without another support; stop this template.
                if local_tick > hold and state.y > block.y+24 and state.vy > 0:
                    break


def cassette_launch_edges(simulator, start, goal, *, deadline=math.inf,
                          max_frames=40, stats=None):
    """Walk to either real support edge, stopping before contact is lost."""
    if start.dead or start.cassette_clock is None or not goal.achieved(start):
        return
    for direction, label in ((LEFT, "left_edge"), (RIGHT, "right_edge")):
        state = start
        actions, states = [], [start]
        for _ in range(min(80, max_frames)):
            if time.monotonic() >= deadline:
                return
            successor = simulator.step(state, direction)
            if stats is not None:
                stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            if successor.dead or successor.room != start.room or not goal.achieved(successor):
                break
            actions.append(direction)
            states.append(successor)
            state = successor
        if actions and abs(state.x-start.x) >= 2:
            remember_segment(start, state, states)
            yield list(actions), state, label
