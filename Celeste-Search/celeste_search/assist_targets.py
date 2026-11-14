"""Selectable room portals, derived from the simulator's native bounds rules.

The extra tile is a transition control, not a fictitious piece of playable map.
Permanent geometry on both sides must admit a player collider. Movers, breakable
walls and hazards are still checked by route search, so an outlined exit is not
a promise that any current player state can reach it.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from copy import deepcopy

from .maps import TILE_SIZE
from .physics import SimState, Simulator, _runtime_height


class _ExitGeometry(Simulator):
    def solid_at(self, room_name, x, y, *, ducking=False, state=None, dynamic_rects=None):
        # A door can become usable after a dashBlock breaks or a mover departs.
        return super().solid_at(room_name, x, y, ducking=ducking, dynamic_rects=())


_CACHE = OrderedDict()


def _arrival_bounds(room, edge, x, y):
    """Collider-centre area immediately after native entry repositioning."""
    if edge == "left":
        return [room.x + room.width - 16, y, room.x + room.width, y + TILE_SIZE]
    if edge == "right":
        return [room.x, y, room.x + 16, y + TILE_SIZE]
    if edge == "top":
        bottom = room.y + _runtime_height(room)
        return [x, bottom - 24, x + TILE_SIZE, bottom]
    return [x, room.y, x + TILE_SIZE, room.y + 24]


def room_exit_targets(map_data, room_name):
    """Return source-room 8px extension tiles with canonical portal metadata.

    ``transition_to`` names the actual neighbor; ``arrival_bounds`` is an
    exclusive-right/bottom collider-centre box in that neighbor. For a legacy
    184px editor room whose runtime height is 180, the bottom extension remains
    on the editor grid at y+184. The 180..183 strip is not playable room space.
    """
    key = (id(map_data), room_name)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] is map_data:
        _CACHE.move_to_end(key)
        return deepcopy(list(cached[1]))
    room = map_data.rooms.get(room_name)
    if room is None or room.dummy:
        return []
    sim = _ExitGeometry(map_data)
    bottom = room.y + _runtime_height(room)
    grid_bottom = room.y + math.ceil(_runtime_height(room) / TILE_SIZE) * TILE_SIZE
    right = room.x + room.width
    candidates = []
    for y in range(room.y, bottom, TILE_SIZE):
        candidates.extend((("left", room.x-TILE_SIZE, y), ("right", right, y)))
    for x in range(room.x, right, TILE_SIZE):
        candidates.extend((("top", x, room.y-TILE_SIZE), ("bottom", x, grid_bottom)))
    result = []
    traversable_lanes = {}
    for edge, tile_x, tile_y in candidates:
        destinations = set()
        for ducking in (False, True):
            height = 6 if ducking else 11
            for lane in range(TILE_SIZE):
                if edge in {"left", "right"}:
                    # Integer feet yield half-pixel collider centres normally.
                    foot_y = tile_y + lane + math.ceil(height/2)
                    foot_x = room.x+3 if edge == "left" else right-3
                    inside_x, inside_y = (room.x+4 if edge == "left" else right-4), foot_y
                elif edge == "top":
                    foot_x, foot_y = tile_x+lane, room.y+math.ceil(height/2)-1
                    inside_x, inside_y = foot_x, room.y+height
                else:
                    foot_x, foot_y = tile_x+lane, bottom+1
                    inside_x, inside_y = foot_x, bottom
                source = SimState(room_name, foot_x, foot_y, ducking=ducking)
                if sim.solid_at(room_name, foot_x, foot_y, ducking=ducking):
                    continue
                # Check the approach corridor too: a thin permanent barrier
                # cannot become an exit merely because the probe is beyond it.
                distance = max(abs(foot_x-inside_x), abs(foot_y-inside_y))
                if any(sim.solid_at(room_name,
                                   inside_x+(foot_x-inside_x)*step/max(1,distance),
                                   inside_y+(foot_y-inside_y)*step/max(1,distance),
                                   ducking=ducking)
                       for step in range(int(distance)+1)):
                    continue
                entered = sim._bounds(source)
                if entered.room == room_name or entered.dead:
                    continue
                if sim.solid_at(entered.room, entered.x, entered.y, ducking=ducking):
                    continue
                destinations.add(entered.room)
                traversable_lanes.setdefault((edge, tile_x, tile_y, entered.room), set()).add(lane)
        # Geometry order matches Simulator._bounds if unusual maps have two
        # destination rooms meeting inside a single source tile.
        for neighbor in sim._neighbors[room_name]:
            if neighbor.name in destinations:
                result.append(dict(room=room_name, x=tile_x, y=tile_y, tile_size=TILE_SIZE,
                                   edge=edge, transition_to=neighbor.name,
                                   arrival_bounds=_arrival_bounds(neighbor,edge,tile_x,tile_y)))
    _CACHE[key] = (map_data, tuple(result), traversable_lanes)
    while len(_CACHE) > 128:
        _CACHE.popitem(last=False)
    return deepcopy(result)


def expand_exit_target(map_data, target):
    """Select the connected portal containing this canonical exterior tile.

    Historical targets without ``exit_region`` stay exact. A new regional
    selection stores individual canonical lane bounds rather than a bounding
    rectangle, so permanent gaps and different neighboring rooms never become
    reachable just by combining adjacent controls. At a rare partially split
    tile, use the traversable component closest to the selected tile centre.
    Actual movement, hazards and temporary solids are still checked by physics.
    """
    keys = ("room", "x", "y", "tile_size", "edge", "transition_to")
    exits = room_exit_targets(map_data, target.get("room"))
    selected = next((item for item in exits
                     if all(target.get(key) == item[key] for key in keys)), None)
    if selected is None:
        raise ValueError("invalid_transition_target")
    cached = _CACHE[(id(map_data), selected["room"])]
    lane_map = cached[2]
    vertical = selected["edge"] in {"left", "right"}
    axis = "y" if vertical else "x"
    available = {}
    for item in exits:
        if (item["edge"], item["transition_to"]) != (selected["edge"], selected["transition_to"]):
            continue
        key = (item["edge"], item["x"], item["y"], item["transition_to"])
        for lane in lane_map[key]:
            available[item[axis] + lane] = item
    choices = [position for position, item in available.items()
               if item["x"] == selected["x"] and item["y"] == selected["y"]]
    anchor = min(choices, key=lambda position: (abs(position + .5 - (selected[axis] + 4)), position))
    first = last = anchor
    while first - 1 in available:
        first -= 1
    while last + 1 in available:
        last += 1
    members = []
    for item in sorted(exits, key=lambda value: value[axis]):
        if (item["edge"], item["transition_to"]) != (selected["edge"], selected["transition_to"]):
            continue
        start, end = max(first, item[axis]), min(last + 1, item[axis] + TILE_SIZE)
        if start >= end:
            continue
        bounds = list(item["arrival_bounds"])
        bounds[1 if vertical else 0] = start
        bounds[3 if vertical else 2] = end
        members.append(dict(x=item["x"], y=item["y"], arrival_bounds=bounds))
    return dict(selected, exit_region=members)


def transition_arrival_bounds(target):
    """Return separate arrival lanes, preserving gaps in an explicit region."""
    members = target.get("exit_region")
    if members is None:
        return (target.get("arrival_bounds"),)
    if not isinstance(members, (tuple, list)) or not members:
        return ()
    return tuple(item.get("arrival_bounds") for item in members if isinstance(item, dict))


def transition_target_reached(state, target):
    """Only the intended next room's entry lane satisfies an exterior tile."""
    if isinstance(state, dict):
        room, x, y, dead = (state.get(k) for k in ("room", "x", "y", "dead"))
        ducking = state.get("ducking", state.get("model_state", {}).get("ducking", False))
    else:
        room, x, y, dead, ducking = state.room, state.x, state.y, state.dead, state.ducking
    if dead or room != target.get("transition_to") or x is None or y is None:
        return False
    center_y = y - (3 if ducking else 5.5)
    return any(isinstance(bounds, (tuple, list)) and len(bounds) == 4
               and all(type(value) in (int, float) and math.isfinite(value) for value in bounds)
               and bounds[0] <= x < bounds[2] and bounds[1] <= center_y < bounds[3]
               for bounds in transition_arrival_bounds(target))
