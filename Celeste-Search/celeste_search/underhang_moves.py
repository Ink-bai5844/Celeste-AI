"""Physical edge-drop/reversal search for targets beneath one-way platforms."""
from __future__ import annotations

import time

from .physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment, search_tracker


def underhang_roofs(simulator, start, goal):
    """Nearby one-way roofs above the selected tile, with attached spike span.

    This is geometry guidance only: every proposed crossing still runs through
    the simulator and must satisfy the exact selected-tile predicate.
    """
    room_name = getattr(goal, "source_room", "") or goal.room
    if (start.dead or start.room != room_name or getattr(goal, "transition_to", "")
            or not hasattr(goal, "tile_x") or not hasattr(goal, "tile_y")):
        return []
    data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
    room = data.rooms[start.room]
    target_x = goal.tile_x+goal.tile_size/2
    roofs = []
    for entity in room.entities:
        if (entity.kind != "jumpThru" or not entity.x <= target_x < entity.x+entity.width
                or not entity.y < goal.tile_y+goal.tile_size
                or abs(entity.x-start.x) > 384 or abs(entity.y-start.y) > 240):
            continue
        left, right = entity.x, entity.x+entity.width
        for spike in room.entities:
            if (spike.kind == "spikesUp" and abs(spike.y-entity.y) < .1
                    and spike.x < entity.x+entity.width
                    and spike.x+max(8, spike.width) > entity.x):
                left = min(left, spike.x)
                right = max(right, spike.x+max(8, spike.width))
        roofs.append(dict(identity=f"{start.room}:{entity.id}", left=left, right=right,
                          top=entity.y, bottom=entity.y+5))
    return sorted(roofs, key=lambda roof: (abs(goal.tile_y-roof["top"]), roof["identity"]))[:4]


def underhang_edges(simulator, start, goal, *, deadline, max_frames=160, stats=None):
    """Yield complete, alive physical routes into the exact tile below a roof.

    Coast outside either edge, reverse while descending into the tile's row,
    then dash inward. Dash starts only after the full movement body clears the
    roof's five-pixel collider; otherwise native jump-through assistance can
    pull the player upward and out of the requested row.
    """
    if max_frames < 1 or time.monotonic() >= deadline:
        return
    roofs = underhang_roofs(simulator, start, goal)
    preview = search_tracker(start)
    expanded = simulated = 0
    seen = set()
    for roof in roofs:
        for outward, inward, edge in ((LEFT, RIGHT, roof["left"]-4),
                                      (RIGHT, LEFT, roof["right"]+4)):
            if time.monotonic() >= deadline:
                return
            if (stats is not None and stats.get("max_expansions") is not None
                    and stats.get("expansions", 0) >= stats["max_expansions"]):
                return
            state = start
            actions, states = [], [state]
            expanded += 1
            if stats is not None:
                stats["expansions"] = stats.get("expansions", 0)+1

            def advance(mask):
                nonlocal state, simulated
                if len(actions) >= max_frames or time.monotonic() >= deadline:
                    return False
                state = simulator.step(state, mask)
                actions.append(mask)
                states.append(state)
                simulated += 1
                if stats is not None:
                    stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
                if preview is not None:
                    preview.report(lambda: [("alternative", state, states)], expanded, simulated)
                return not state.dead and state.room == start.room

            valid = True
            # Coordinates are tested on actual simulated states. Starting below
            # or away from a roof is allowed: coasting may first use a spring or
            # land on a crumble platform before reaching the outside edge.
            while not goal.achieved(state) and not (state.x <= edge if outward == LEFT else state.x >= edge):
                if not advance(outward):
                    valid = False
                    break
            while valid and not goal.achieved(state):
                height = 6 if state.ducking else 11
                centre_y = state.y-height/2
                if (state.y-height >= roof["bottom"]
                        and goal.tile_y <= centre_y < goal.tile_y+goal.tile_size):
                    break
                if not advance(inward):
                    valid = False
                    break
            if not valid:
                continue
            if not goal.achieved(state):
                if state.prev_input & DASH and not advance(inward):
                    continue
                for _ in range(min(18, max_frames-len(actions))):
                    if goal.achieved(state):
                        break
                    if not advance(inward | DASH):
                        valid = False
                        break
            if not valid or not goal.achieved(state) or not actions or state in seen:
                continue
            seen.add(state)
            label = f"underhang:{roof['identity']}:{'left' if outward == LEFT else 'right'}"
            remember_segment(start, state, states)
            if preview is not None:
                preview.report(lambda: [("best", state, states)], expanded, simulated, remember=True)
            yield actions, state, label


def route_has_underhang(route, state, map_data):
    """Cheap geometry-only portfolio hint for a nearby selected tile below a roof."""
    if not route or state is None or map_data is None:
        return False
    room_name = state.get("room") if isinstance(state, dict) else state.room
    room = map_data.rooms.get(room_name)
    if room is None:
        return False
    for point in route.get("waypoints", [route]):
        if point.get("room") != room_name or point.get("transition_to"):
            continue
        size = point.get("tile_size", 8)
        x, y = point["x"] + size/2, point["y"]
        if any(entity.kind == "jumpThru" and entity.x <= x < entity.x+entity.width
               and entity.y < y+size <= entity.y+72 for entity in room.entities):
            return True
    return False


def _roof_side_barriers(simulator, start, goal):
    """Find exposed solid walls attached to a roof, including adjoining floor tiles."""
    from .wall_moves import _static_faces
    data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
    room = data.rooms[start.room]
    tiles = simulator._tiles[start.room]
    faces = list(_static_faces(simulator, room))
    for roof in underhang_roofs(simulator, start, goal):
        left, right, top = roof["left"], roof["right"], roof["top"]
        row = int((top-room.y)//8)
        while (int((left-room.x)//8)-1, row) in tiles:
            left -= 8
        while (int((right-room.x)//8), row) in tiles:
            right += 8
        for side, edge, outward, inward in ((1, left, LEFT, RIGHT), (-1, right, RIGHT, LEFT)):
            if not room.x+4 < edge < room.x+room.width-4:
                continue
            for rect, facing in faces:
                face = rect[0] if facing > 0 else rect[2]
                if (facing == side and abs(face-edge) < .1 and rect[1] <= top < rect[3]
                        and top+8 < rect[3] <= top+128
                        and goal.tile_y+goal.tile_size <= rect[3]+11):
                    yield roof, edge, rect[3], outward, inward


def underhang_detour_edges(simulator, start, goal, *, deadline, max_frames=220,
                           stats=None, allow_dash=True):
    """Drop around an adjoining solid wall, recover below the roof, then rise.

    A roof's immediate edge can itself stand on a solid wall or lower platform.
    Merely reversing at that edge never leaves the floor. Enumerate small extra
    clearances, let real collisions determine the descent, and turn toward the
    original tile. A late upward dash is a fallback after an ordinary climb
    attempt; a recovered support starts a real jump with its actual resources.
    No waypoint, position, velocity, mechanism phase or dash count is invented.
    """
    if start.dead or max_frames < 1 or time.monotonic() >= deadline:
        return
    seen = set()
    preview = search_tracker(start)
    for roof, edge, bottom, outward, inward in _roof_side_barriers(simulator, start, goal):
        target_x = goal.tile_x+goal.tile_size/2
        for boost in ((False, True) if allow_dash else (False,)):
            for clearance in (4, 8, 0, 12, 16):
                if (time.monotonic() >= deadline or stats is not None and
                        stats.get("max_expansions") is not None and
                        stats.get("expansions", 0) >= stats["max_expansions"]):
                    return
                if stats is not None:
                    stats["expansions"] = stats.get("expansions", 0)+1
                state, actions, states = start, [], [start]
                outside = edge-4-clearance if outward == LEFT else edge+4+clearance

                def advance(mask):
                    nonlocal state
                    if len(actions) >= max_frames or time.monotonic() >= deadline:
                        return False
                    state = simulator.step(state, mask)
                    actions.append(mask)
                    states.append(state)
                    if stats is not None:
                        stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
                    return not state.dead and state.room == start.room

                valid = True
                while not goal.achieved(state) and (state.x > outside if outward == LEFT else state.x < outside):
                    if not advance(outward):
                        valid = False
                        break
                # Walking inward can press against the wall while gravity
                # clears its bottom. A lower ledge can stop the descent; that
                # branch simply exhausts its bounded recipe and is discarded.
                while valid and not goal.achieved(state) and (state.x < target_x if inward == RIGHT else state.x > target_x):
                    if not advance(inward):
                        valid = False
                        break
                if not valid:
                    continue
                if not goal.achieved(state):
                    if boost:
                        if not state.dashes or not advance(DASH | UP):
                            continue
                        # Dash direction is sampled on the following update.
                        coast_mask = UP
                    else:
                        coast_mask = GRAB | UP | outward
                    for _ in range(90):
                        if goal.achieved(state):
                            break
                        if not advance(coast_mask):
                            valid = False
                            break
                        if state.on_ground and state.dashes:
                            break
                if valid and not goal.achieved(state) and state.on_ground:
                    for _ in range(48):
                        if goal.achieved(state):
                            break
                        if not advance(JUMP):
                            valid = False
                            break
                if not valid or not actions or not goal.achieved(state) or state in seen:
                    continue
                seen.add(state)
                label = f"underhang_detour:{roof['identity']}:{'left' if outward == LEFT else 'right'}:{clearance}:{int(boost)}"
                remember_segment(start, state, states)
                if preview is not None:
                    preview.report(lambda: [("best", state, states)],
                                   (stats or {}).get("expansions", 0),
                                   (stats or {}).get("simulated_frames", 0), remember=True)
                yield actions, state, label


def underhang_climb_macros(simulator, state, goal):
    """Keep a wall ascent together even when a dash is still available."""
    if state.dead or state.on_ground or goal.room != state.room or goal.y >= state.y or state.stamina <= 0:
        return
    for direction, button in ((-1, LEFT), (1, RIGHT)):
        if any(simulator._wall(state.room, state.x, state.y, direction, distance)
               for distance in (1, 4, 8)):
            for duration in (12, 24, 48, 72):
                yield GRAB | UP | button, duration
