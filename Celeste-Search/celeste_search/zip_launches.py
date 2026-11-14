"""Geometry-guided, fully simulated launches from moving zip platforms."""
from __future__ import annotations

import time

from .physics import DASH, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment, search_tracker


def zip_launch_edges(simulator, start, board, goal, *, deadline,
                     max_frames=160, stats=None):
    """Yield exact target hits reached by real lift-assisted launch branches.

    A rising platform beneath an overhang can require a horizontal dash to its
    edge before jumping. Grounded dash refill and lift inheritance are produced
    by ordinary simulator updates, never assigned by this helper. Reversing
    into the target column before an upward dash keeps narrow shafts usable.
    """
    source = getattr(goal, "source_room", "") or goal.room
    if (start.dead or start.room != source or not board.achieved(start)
            or max_frames < 1 or time.monotonic() >= deadline):
        return
    tile_x = getattr(goal, "tile_x", goal.x - 4)
    size = getattr(goal, "tile_size", 8)
    centre = tile_x + size / 2
    platform_x, _ = board.location(start)
    direction = 1 if centre >= start.x else -1
    outward, inward = (RIGHT, LEFT) if direction == 1 else (LEFT, RIGHT)
    far_edge = platform_x + board.width if direction == 1 else platform_x
    # The offsets are body-clearance distances, independent of room coordinates
    # and of how many frames the platform has already been moving.
    branches = [("aligned", 0, 0, False)]
    branches += [("reverse", clearance, excursion, held)
                 for clearance in (4, 8, 0)
                 for excursion in (16, 8, 24, 32)
                 for held in (False, True)]
    preview = search_tracker(start)
    expanded = simulated = 0
    seen = set()
    for kind, clearance, excursion, held in branches:
        if time.monotonic() >= deadline:
            return
        if (stats is not None and stats.get("max_expansions") is not None
                and stats.get("expansions", 0) >= stats["max_expansions"]):
            return
        expanded += 1
        if stats is not None:
            stats["expansions"] = stats.get("expansions", 0) + 1
        state = start
        actions, states = [], [start]
        hit = False

        def advance(mask):
            nonlocal state, simulated, hit
            if len(actions) >= max_frames or time.monotonic() >= deadline:
                return False
            previous = state
            state = simulator.step(previous, mask)
            actions.append(mask)
            states.append(state)
            simulated += 1
            if stats is not None:
                stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
            hit = not state.dead and goal.achieved(state)
            destination = getattr(goal, "transition_to", "")
            if destination:
                hit = (hit and previous.room == source and state.room == destination)
            if preview is not None:
                preview.report(lambda: [("alternative", state, states)], expanded, simulated)
            return not state.dead and (state.room == source or hit)

        valid = True
        if kind == "aligned":
            # An already aligned rider can keep the lift without horizontal
            # overshoot. Walking into alignment is itself simulated and may be
            # rejected by an approaching ceiling or a changing platform.
            while not tile_x <= state.x < tile_x + size:
                if not advance(RIGHT if state.x < tile_x else LEFT):
                    valid = False
                    break
            if valid and state.prev_input & JUMP:
                valid = advance(0)
            for tick in range(36):
                if not valid or hit:
                    break
                if tick and state.vy >= -30:
                    break
                valid = advance(JUMP | UP)
        else:
            launch_x = far_edge - direction * clearance
            if state.prev_input & DASH:
                valid = advance(outward)
            while valid and not hit and direction * (state.x - launch_x) < 0:
                valid = advance(outward | DASH)
            # Jump only from a real support or its remaining coyote grace.
            if valid and not hit and not (state.on_ground or state.grace > 0):
                valid = False
            excursion_x = centre + direction * excursion
            for tick in range(32):
                if not valid or hit or direction * (state.x - excursion_x) >= 0:
                    break
                previous_x = state.x
                valid = advance(outward | JUMP)
                if tick > 1 and state.x == previous_x and abs(state.vx) < 1:
                    break
            # Enter the selected column from the far side; the native wall
            # collision and real horizontal acceleration determine this frame.
            while valid and not hit and not tile_x <= state.x < tile_x + size:
                valid = advance(inward | (JUMP if held else 0))
        if not valid:
            continue
        if not hit:
            if state.prev_input & DASH and not advance(UP):
                continue
            for _ in range(30):
                if hit:
                    break
                if not advance(UP | DASH):
                    valid = False
                    break
        if not valid or not hit or not actions or state in seen:
            continue
        seen.add(state)
        label = f"zip_launch:{board.identity}:{kind}:{direction}:{clearance}:{excursion}:{int(held)}"
        remember_segment(start, state, states)
        if preview is not None:
            preview.report(lambda: [("best", state, states)], expanded, simulated, remember=True)
        yield actions, state, label
