"""Weighted graph search over input sequences in the offline simulator.

No neural network, fitting, learned policy, sockets, or game process is used here.
The search is bounded and heuristic: a budget exhaustion is never called impossible.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, replace
import heapq
import math
import time
from typing import Any
from .search_progress import search_tracker


@dataclass(frozen=True)
class Goal:
    kind: str
    room: str
    x: float
    y: float
    id: str = ""
    no_dash_until: str = ""

    @property
    def key(self):
        return self.id or f"{self.kind}:{self.room}:{int(self.x)}:{int(self.y)}"

    def achieved(self, state) -> bool:
        if state.dead:
            return False
        if self.kind == "berry":
            return self.id in state.berries
        if self.kind == "break":
            return self.id.removeprefix("break:") in state.broken
        if self.kind == "room":
            return state.room == self.room
        if self.kind == "surface":
            return (state.room == self.room and state.on_ground
                    and abs(state.x-self.x) < 12 and abs(state.y-self.y) < 3)
        return state.room == self.room and math.hypot(state.x - self.x, state.y - self.y) < 12


def berry_id(room: str, entity) -> str:
    return f"{room}:{entity.id}"


def ordinary_berries(room):
    return [e for e in room.entities if e.kind.lower() == "strawberry"
            and not e.attrs.get("golden", False) and not e.attrs.get("moon", False)
            and not e.nodes]


def winged_berries(room):
    return [e for e in ordinary_berries(room)
            if e.attrs.get("winged") is True or str(e.attrs.get("winged")).lower() in {"true","1"}]


def secret_goals(room):
    """Explicit destructible entities and visual hidden regions are distinct."""
    goals=[]
    for e in room.entities:
        if e.kind == "dashBlock" and e.attrs.get("canDash",True) is not False:
            goals.append(Goal("break",room.name,e.x+e.width/2,e.y+e.height/2+5,
                              f"break:{room.name}:{e.id}"))
        elif e.kind in {"fakeWall","coverupWall"}:
            goals.append(Goal("hidden",room.name,e.x+e.width/2,e.y+e.height/2+5,
                              f"hidden:{room.name}:{e.id}"))
    return goals


def room_is_supported(simulator, name: str, state=None) -> bool:
    observed_cassette = (state is not None and (getattr(state,"cassette_clock",None) is not None
                                              or getattr(state,"cassette_stopped",False)))
    report = simulator.capability_report(name, state=state) if observed_cassette else simulator.capability_report(name)
    return not report.get("unsupported", report.get("unsupported_entities", []))


def terminal_goal_crossing_allowed(simulator,goal,before,after):
    """An unsupported room is allowed only as an inspected terminal crossing."""
    certificate=getattr(goal,"terminal_entry",None)
    if certificate is None or not goal.achieved(after):
        return False
    from .terminal_entry import terminal_entry_crossing_allowed
    return terminal_entry_crossing_allowed(simulator,certificate,before,after)


class DistanceField:
    """Optimistic geometry heuristic. It is not a claim of physical reachability."""
    def __init__(self, room, x, y):
        self.room = room
        self.width = (room.width + 7) // 8
        self.height = (room.height + 7) // 8
        self.distance = {}
        # Hazard-aware guidance still does not constrain physical transitions.
        # Ignoring spike tops steers every branch toward the same lethal shortcut
        # instead of the narrow path beneath a staggered one-way platform.
        self.blocked=set()
        for entity in room.hazard_entities:
            if entity.kind=="spikesUp":
                rect=(entity.x-4,entity.y-9,entity.x+max(8,entity.width)+4,entity.y+6)
            elif entity.kind=="spikesDown":
                rect=(entity.x-4,entity.y-6,entity.x+max(8,entity.width)+4,entity.y+9)
            elif entity.kind=="spikesLeft":
                rect=(entity.x-9,entity.y-6,entity.x+4,entity.y+max(8,entity.height)+6)
            elif entity.kind=="spikesRight":
                rect=(entity.x-4,entity.y-6,entity.x+9,entity.y+max(8,entity.height)+6)
            else:
                continue
            for a in range(max(0,int((rect[0]-room.x)//8)),min(self.width,int((rect[2]-room.x)//8)+1)):
                for b in range(max(0,int((rect[1]-room.y)//8)),min(self.height,int((rect[3]-room.y)//8)+1)):
                    if rect[0]<=room.x+a*8+4<rect[2] and rect[1]<=room.y+b*8+4<rect[3]:
                        self.blocked.add((a,b))
        tx, ty = int((x - room.x) // 8), int((y - room.y - 5) // 8)
        candidates = [(abs(a - tx) + abs(b - ty), a, b)
                      for a in range(max(0, tx - 3), min(self.width, tx + 4))
                      for b in range(max(0, ty - 3), min(self.height, ty + 4))
                      if not room.tile_at(a, b) and (a,b) not in self.blocked]
        if not candidates:
            return
        _, tx, ty = min(candidates)
        queue = deque([(tx, ty)])
        self.distance[(tx, ty)] = 0
        while queue:
            a, b = queue.popleft()
            cost = self.distance[(a, b)] + 8
            for p, q in ((a-1,b), (a+1,b), (a,b-1), (a,b+1)):
                if (0 <= p < self.width and 0 <= q < self.height
                        and (p,q) not in self.distance and not room.tile_at(p,q)
                        and (p,q) not in self.blocked):
                    self.distance[(p,q)] = cost
                    queue.append((p,q))

    def at(self, x, y):
        return self.distance.get((int((x-self.room.x)//8), int((y-self.room.y-5)//8)))


class Heuristic:
    def __init__(self, map_data, goal: Goal):
        self.map_data, self.goal = map_data, goal
        self.fields = {}
        self.supports = {}
        self.parents = {goal.room: None}
        incoming = {name: [] for name in map_data.rooms}
        for name in map_data.rooms:
            for neighbor in map_data.neighbors(name):
                incoming[neighbor.name].append(name)
        queue = deque([goal.room])
        while queue:
            current = queue.popleft()
            for neighbor in incoming[current]:
                if neighbor not in self.parents:
                    self.parents[neighbor] = current
                    queue.append(neighbor)

    def waypoint(self, state):
        goal = self.goal
        room = self.map_data.rooms[state.room]
        dynamic=getattr(goal,"waypoint",None)
        if state.room==goal.room and callable(dynamic):
            return dynamic(state)
        if goal.kind == "berry" and goal.id in state.following:
            if state.room not in self.supports:
                self.supports[state.room]=coverage_goals(room)
            floors=[(math.hypot(state.x-g.x,state.y-g.y),g.x,g.y)
                    for g in self.supports[state.room]]
            if floors:
                _, x, y = min(floors)
                return x, y
        if state.room == goal.room:
            return goal.x, goal.y
        next_name = self.parents.get(state.room)
        if next_name is None:
            return goal.x, goal.y
        other = self.map_data.rooms[next_name]
        if room.x + room.width <= other.x:
            return room.x+room.width+5, min(max(state.y,max(room.y,other.y)+16), min(room.y+room.height,other.y+other.height)-8)
        if other.x + other.width <= room.x:
            return room.x-5, min(max(state.y,max(room.y,other.y)+16), min(room.y+room.height,other.y+other.height)-8)
        x = min(max(state.x,max(room.x,other.x)+8), min(room.x+room.width,other.x+other.width)-8)
        return x, room.y-5 if other.y < room.y else room.y+room.height+5

    def __call__(self, state):
        x, y = self.waypoint(state)
        field_key = (state.room, int(x//8), int(y//8))
        if field_key not in self.fields:
            if len(self.fields)>128:
                self.fields.clear()
            self.fields[field_key] = DistanceField(self.map_data.rooms[state.room], x, y)
        field_distance = self.fields[field_key].at(state.x,state.y)
        distance = math.hypot(state.x-x, state.y-y)
        if field_distance is not None:
            distance = max(distance, field_distance)
        if self.goal.kind == "berry" and self.goal.id in state.following:
            # Picking up a berry is a distinct milestone: returning to ground
            # must remain preferable to approaching the pickup position again.
            return distance * 0.4 - 150 + (0 if state.on_ground else 8)
        # Account for every remaining room, without assuming adjacency means reachable.
        name, depth = state.room, 0
        while name != self.goal.room and name in self.parents and depth < 100:
            name = self.parents[name]
            depth += 1
        if depth:
            # Retain geometry costs in EVERY remaining room. A fixed charge
            # per room falsely rewards stepping backward out of a difficult
            # room: its long underhang path disappears from the estimate.
            total=distance
            name=self.parents.get(state.room)
            px,py=x,y
            for _ in range(depth):
                if name is None:
                    break
                probe=replace(state,room=name,x=px,y=py)
                tx,ty=self.waypoint(probe)
                key=(name,int(tx//8),int(ty//8))
                if key not in self.fields:
                    self.fields[key]=DistanceField(self.map_data.rooms[name],tx,ty)
                remaining=self.fields[key].at(px,py)
                total+=max(math.hypot(px-tx,py-ty),remaining or 0)
                if name==self.goal.room:
                    break
                name=self.parents.get(name)
                px,py=tx,ty
            return total
        return distance


def state_key(state):
    """Deliberate discretization for search only; simulation retains full state."""
    return (state.room, round(state.x/3), round(state.y/3), round(state.vx/30),
            round(state.vy/30), state.dashes, int(state.stamina/22), state.on_ground,
            state.state, getattr(state,"prev_input",0) & 112,
            round(getattr(state,"dash_frames",0)/2),
            getattr(state,"dash_attack",0),getattr(state,"wall_retention",0),
            round(getattr(state,"wall_retained",0)/30),
            getattr(state,"hop_wait_x",0),round(getattr(state,"hop_wait_x_speed",0)/30),
            getattr(state,"climb_hop_solid",""),getattr(state,"climb_hop_x",0.),
            getattr(state,"climb_hop_y",0.),
            round(getattr(state,"var_jump",0)/2),
            round(getattr(state,"grace",0)/2),
            state.berries, state.following, getattr(state,"fled",frozenset()),
            getattr(state,"collect_delay",0),getattr(state,"collect_ground",0),
            getattr(state,"dash_cooldown",0),getattr(state,"refill_cooldown",0),
            getattr(state,"refill_timers",()),getattr(state,"broken",frozenset()),
            getattr(state,"mechanisms",()),getattr(state,"lift_timer",0),
            round(getattr(state,"lift_vx",0)/30),round(getattr(state,"lift_vy",0)/30),
            getattr(state,"force_x",0),getattr(state,"force_frames",0),
            getattr(state,"ducking",False),getattr(state,"facing",1),
            getattr(state,"auto_jump_active",False),getattr(state,"auto_jump",0),
            getattr(state,"climb_wait",0),getattr(state,"jump_buffer",0),
            getattr(state,"dash_buffer",0),getattr(state,"follow_order",()),
            getattr(state,"cassette_stopped",False),
            getattr(state,"cassette_clock",None),getattr(state,"cassette_blocks",()),
            getattr(state,"player_depth",0),getattr(state,"player_actual_depth",0.0))


def _underhang_sequences(simulator,state,goal):
    """Physical edge-drop/reverse rollouts for one-way-platform underpasses."""
    if not state.on_ground or goal.kind!="passage":
        return
    room=(simulator.map_data if hasattr(simulator,"map_data") else simulator.map).rooms[state.room]
    platforms=[e for e in room.entities if e.kind=="jumpThru"
               and f"under:{room.name}:{e.id}"==goal.id and abs(e.y-state.y)<3
               and e.x-4<=state.x<=e.x+e.width+4]
    for platform in platforms:
        for outward,inward in ((1,2),(2,1)):
            current=state
            inputs=[]
            for _ in range(42):
                current=simulator.step(current,outward)
                inputs.append(outward)
                if current.dead or current.room!=state.room or not current.on_ground:
                    break
            if current.dead or current.room!=state.room or current.on_ground:
                continue
            for _ in range(30):
                current=simulator.step(current,inward)
                inputs.append(inward)
                if (current.dead or current.room!=state.room or
                        (platform.x+4<=current.x<=platform.x+platform.width-4
                         and current.y>=platform.y+10)):
                    break
            if current.dead or current.room!=state.room:
                continue
            for _ in range(60):
                if goal.achieved(current):
                    break
                current=simulator.step(current,0)
                inputs.append(0)
                if current.dead or current.room!=state.room or current.on_ground:
                    break
            if not current.dead and current.room==state.room:
                yield tuple(inputs),len(inputs)


def macros(state, heuristic, variant=0, simulator=None):
    if getattr(heuristic.goal,"handoff_frames",0):
        from .safe_handoff import target_approach_macros
        yield from target_approach_macros(state,heuristic.goal)
    if heuristic.goal.kind=="wall_grip":
        from .wall_moves import wall_approach_macros
        yield from wall_approach_macros(state,heuristic.goal)
    if simulator is not None:
        yield from _underhang_sequences(simulator,state,heuristic.goal)
    x, y = heuristic.waypoint(state)
    direction = 1 if x >= state.x else 2
    opposite = 2 if direction == 1 else 1
    # Holding and releasing buttons are separate edges (jump/dash are edge-triggered).
    actions = [direction, direction|16, 0, 16, opposite, opposite|16,
               direction|64, opposite|64, 64|4, 64|16|direction, 64|16|opposite]
    protected=heuristic.goal.no_dash_until
    if state.dashes and (not protected or protected in state.following or protected in state.berries):
        vertical = 4 if y < state.y - 8 else 8 if y > state.y + 16 else 0
        actions += [32|direction|vertical,32|4,32|direction,32|opposite,
                    32|opposite|4,32|direction|4,32|8,32|direction|8,32|opposite|8]
    durations = ((6,12),(4,10),(3,8),(8,16),(2,6),(5,14))[variant%6]
    if variant>=6:
        rotation=(variant//6)%len(actions)
        actions=actions[rotation:]+actions[:rotation]
    seen = set()
    if state.following:
        for duration in (12,30,60):
            seen.add((0,duration))
            yield 0,duration
    for action in actions:
        for duration in (((4,8),(3,6),(2,7))[variant%3] if action & 32 else durations):
            if (action,duration) not in seen:
                seen.add((action,duration))
                yield action,duration


@dataclass
class SearchResult:
    actions: list[int]
    state: Any
    reached: bool
    expansions: int
    simulated_frames: int
    reason: str


@dataclass
class SearchNodeBudget:
    """Shared live-node ceiling for explicitly suspended searches."""
    max_nodes: int = 80000
    nodes: int = 0


class SearchContinuation:
    """Caller-owned search state, independent of the opportunistic global LRU."""
    def __init__(self, node_budget=None):
        self.node_budget = node_budget or SearchNodeBudget()
        if type(self.node_budget.max_nodes) is not int or self.node_budget.max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        self.session = None
        self.key = None
        self.nodes = 0
        self.resumes = 0

    def claim(self):
        if self.node_budget.nodes >= self.node_budget.max_nodes:
            return False
        self.node_budget.nodes += 1
        self.nodes += 1
        return True

    def release(self):
        self.node_budget.nodes -= self.nodes
        self.nodes = 0
        self.session = None
        self.key = None



@dataclass(frozen=True)
class GroundArrival:
    """A safe explored ground prefix; it does not assert public-goal success."""
    actions: tuple[int, ...]
    state: Any
    support_key: tuple


def _ground_static_spans(simulator, room_name, height):
    """Cache exposed, contiguous permanent supports by collider clearance."""
    base = simulator
    while hasattr(base, "base") and base.base is not base:
        base = base.base
    cache = getattr(base, "_search_ground_supports", None)
    if cache is None:
        cache = base._search_ground_supports = {}
    cache_key = (room_name, height)
    if cache_key in cache:
        return cache[cache_key]
    solids = getattr(base, "_static_rects", {}).get(room_name, ())
    platforms = getattr(base, "_platforms", {}).get(room_name, ())
    surfaces = {}
    for left, top, right, _ in (*solids, *platforms):
        surfaces.setdefault(top, []).append((left, right))
    result = []
    for y, spans in surfaces.items():
        merged = []
        for left, right in sorted(spans):
            if merged and left <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(right, merged[-1][1])
            else:
                merged.append((left, right))
        # Support only requires actual foot overlap, including a real edge
        # perch. Permanent head-height walls split otherwise connected floors.
        clear = [(left-4, right+4) for left, right in merged]
        for left, top, right, bottom in solids:
            if not top < y or not bottom > y-height:
                continue
            blocked_left, blocked_right = left-4, right+4
            following = []
            for a, b in clear:
                if blocked_right <= a or blocked_left >= b:
                    following.append((a, b))
                else:
                    if blocked_left > a:
                        following.append((a, blocked_left))
                    if blocked_right < b:
                        following.append((blocked_right, b))
            clear = following
        result.extend((left, y, right) for left, right in clear if left < right)
    cache[cache_key] = tuple(result)
    return cache[cache_key]


def _ground_hazard_zones(simulator, state, left, right, y):
    """Stable island identity; actual alive physics decides whether it is safe.

    Hazard bounds are conservative (especially spinner corners), so they split
    identity instead of discarding an already observed safe player position.
    Moving attached spikes use their current pose but stable hazard id/side.
    """
    from .physics import _spike_rect
    records = {row[0]: row for row in state.mechanisms}
    attachments = getattr(simulator, "_spike_attachments", {}).get(state.room, {})
    height = 6 if state.ducking else 11
    zones = []
    for hazard in getattr(simulator, "_hazards", {}).get(state.room, ()):
        x, hy = hazard.x, hazard.y
        mover = attachments.get((hazard.kind, hazard.id))
        if mover is not None:
            identity = f"{state.room}:{mover.id}"
            record = records.get(identity)
            if identity in state.broken or record and mover.kind == "fallingBlock" and record[1] == 4:
                continue
            if record:
                x += round(record[3])-mover.x
                hy += round(record[4])-mover.y
        l, top, r, bottom = ((x-6, hy-6, x+6, hy+6) if hazard.kind == "spinner"
                             else _spike_rect(hazard, x, hy))
        if top >= y-2 or bottom <= y-height or r+4 <= left or l-4 >= right:
            continue
        side = -1 if state.x <= l-4 else 1 if state.x >= r+4 else 0
        zones.append((hazard.kind, hazard.id, side))
    return tuple(zones)


def ground_support_key(simulator, state):
    """Identify the actual grounded support without changing player state.

    Permanent continuous floors use a stable exposed span. Moving/cassette
    platforms use their identity, independent of their current world position.
    An OnGround flag with no present geometric support is not a candidate.
    """
    if state.dead or not state.on_ground or state.vy < 0:
        return None
    dynamic = getattr(simulator, "_mechanism_rects", lambda room, state: ())
    for identity, _, (left, top, right, _) in dynamic(state.room, state):
        if state.x+4 > left and state.x-4 < right and state.y <= top <= state.y+1.01:
            key = ("mechanism", state.room, identity)
            zones = _ground_hazard_zones(simulator, state, left-4, right+4, top)
            return key + (("hazard_zones", zones),) if zones else key
    solids = getattr(simulator, "_static_rects", {}).get(state.room, ())
    platforms = getattr(simulator, "_platforms", {}).get(state.room, ())
    present = any(state.x+4 > left and state.x-4 < right and state.y <= top <= state.y+1.01
                  for left, top, right, _ in (*solids, *platforms))
    if present:
        for left, y, right in _ground_static_spans(simulator, state.room, 6 if state.ducking else 11):
            if left <= state.x <= right and state.y <= y <= state.y+1.01:
                key = ("static", state.room, left, y, right)
                zones = _ground_hazard_zones(simulator, state, left, right, y)
                return key + (("hazard_zones", zones),) if zones else key
    return None


def _ground_arrival_band(simulator, state, support):
    from .search_resources import grip_resource_band
    relative_x = state.x
    if support[0] == "mechanism":
        dynamic = getattr(simulator, "_mechanism_rects", lambda room, state: ())
        for identity, _, rect in dynamic(state.room, state):
            if identity == support[2]:
                relative_x -= rect[0]
                break
    clock = state.cassette_clock
    mechanisms = tuple((row[0], row[1], row[2]//12, round(row[3]/16), round(row[4]/16))
                       for row in state.mechanisms)
    cassette = (None if clock is None else
                (clock.current_index, clock.beat_index, round(clock.beat_timer*12), clock.active))
    resources = (state.dashes, grip_resource_band(state.stamina), state.broken,
                 state.berries, state.following, state.fled,
                 tuple((key, timer//12) for key, timer in state.refill_timers))
    band = (int(relative_x//24), round(state.vx/30), state.state, state.ducking,
            state.prev_input & 112, resources, mechanisms, cassette)
    return band, relative_x


def _retain_ground_record(bucket, band, record, limit):
    """Keep each support's spatial/resource/timing diversity independently."""
    old = bucket.get(band)
    if old is not None and old[0].frame <= record[0].frame:
        # A later point in the same coarse x-band can be the actual platform
        # edge. Keep reached extremes instead of freezing at the band's entry.
        positions = [value[4] for value in bucket.values()]
        if limit == 1 or min(positions) <= record[4] <= max(positions):
            return
        if old[0].frame == min(value[0].frame for value in bucket.values()):
            bucket[band+("first",)] = old
    bucket[band] = record
    if len(bucket) <= limit:
        return
    rows = list(bucket.items())
    earliest = min(rows, key=lambda row: row[1][0].frame)
    selected = [earliest]
    # Preserve the actual ends reached on the support, even when farther from
    # the final goal. Goal-distance pruning would defeat these continuation seeds.
    for entry in (min(rows, key=lambda row: row[1][4]),
                  max(rows, key=lambda row: row[1][4])):
        if len(selected) < limit and entry[0] not in {row[0] for row in selected}:
            selected.append(entry)
    while len(selected) < limit:
        remaining = [row for row in rows if row[0] not in {item[0] for item in selected}]
        if not remaining:
            break
        def diversity(row):
            band, value = row
            return min((200*(band[5:8] != other_band[5:8])
                        + 40*(band[1:5] != other_band[1:5])
                        + min(160, abs(value[4]-other_value[4])))
                       for other_band, other_value in selected), -value[0].frame
        selected.append(max(remaining, key=diversity))
    bucket.clear()
    bucket.update(selected)


# Persistent worker requests continue the same offline frontier from the same
# complete physical anchor. This is not a learned policy and never merges
# states with different inventory, mechanisms, timers or subpixel remainders.
_SEARCH_SESSIONS = OrderedDict()


def resume_search_alternatives(simulator,start):
    """A rejected completed route must resume its frontier, not repeat forever."""
    identity=getattr(simulator,"_planning_identity",None)
    normalized=replace(start,frame=0,reason="")
    for key,session in list(_SEARCH_SESSIONS.items()):
        if key[0] is identity and key[1]==normalized and session[-1]:
            heuristic,nodes,frontier,costs,_,_,_=session
            _SEARCH_SESSIONS[key]=(heuristic,nodes,frontier,costs,0,heuristic(normalized),False)


def search_leg(simulator, start, goal: Goal, *, deadline: float,
               max_expansions=5000, max_frames=900, variant=0,
               allow_unsupported=False, ground_arrivals=None,
               max_ground_arrivals_per_support=8, continuation=None) -> SearchResult:
    """Search one goal, optionally exporting safe explored ground prefixes.

    ``ground_arrivals`` is an output list of GroundArrival records. Each support
    independently retains up to ``max_ground_arrivals_per_support`` candidates;
    distant supports are never displaced by a global goal-distance ranking.
    Collection includes intermediate macro states whose later continuation dies
    or loses to search deduplication. It never changes normal search decisions.
    An optional caller-owned ``continuation`` retains the frontier across small
    quanta without relying on the global LRU. Handles may share a node budget;
    callers release them in finally, and memory exhaustion is a budget result.
    """
    if ground_arrivals is not None and (type(max_ground_arrivals_per_support) is not int
                                        or max_ground_arrivals_per_support < 1):
        raise ValueError("max_ground_arrivals_per_support must be a positive integer")
    # Variants are search recipes, not an ever-increasing greediness schedule.
    # Cycling lets repeated background requests resume the same stored frontier.
    variant%=12
    preview=search_tracker(start)
    first_hit_rejected=getattr(goal,"first_hit_rejected",lambda state:False)
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    if goal.kind == "berry":
        winged={berry_id(goal.room,e) for e in winged_berries(data.rooms[goal.room])}
        if goal.id in winged and goal.id not in start.berries:
            if goal.id in start.fled:
                return SearchResult([],start,False,0,0,"winged_berry_fled_current_room_respawn_needed")
            goal=replace(goal,no_dash_until=goal.id)
    original_frame=start.frame
    start=replace(start,frame=0,reason="")
    if not hasattr(simulator,"_planning_identity"):
        simulator._planning_identity=object()
    archive_key=(simulator._planning_identity,start,goal,variant,max_frames,allow_unsupported)
    if continuation is None:
        session=_SEARCH_SESSIONS.pop(archive_key,None)
    else:
        if continuation.key != archive_key:
            continuation.release()
            continuation.key = archive_key
        session = continuation.session
        if session is not None:
            continuation.resumes += 1
    if session:
        heuristic,nodes,frontier,best_cost,best_index,best_h,reached=session
    else:
        if continuation is not None and not continuation.claim():
            return SearchResult([],replace(start,frame=original_frame),False,0,0,"continuation_memory_budget")
        heuristic = Heuristic(simulator.map_data if hasattr(simulator,"map_data") else simulator.map, goal)
        nodes = [(start,-1,0,0,())]
        initial_h = heuristic(start)
        frontier = [(initial_h*1.5,0)]
        best_cost = {state_key(start):0}
        best_index, best_h = 0, initial_h
        reached = goal.achieved(start)
    expanded = simulated = 0
    memory_limited = False
    def preview_paths():
        choices=[("best",best_index)]
        if frontier and frontier[0][1]!=best_index:
            choices.append(("alternative",frontier[0][1]))
        for kind,index in choices:
            end=nodes[index][0];runs=[]
            while nodes[index][1]>=0:
                node=nodes[index]
                runs.append(list(node[4] if len(node)>4 else ())+[node[0]])
                index=node[1]
            yield kind,end,[nodes[0][0]]+[state for run in reversed(runs) for state in run]
    reason = "goal" if reached else "budget"
    support_cache = {}
    ground_records = {} if ground_arrivals is not None else None

    def collect_ground(state, parent, mask, count):
        if state.dead or not state.on_ground or state.vy < 0 or first_hit_rejected(state):
            return
        if not allow_unsupported:
            if state.room not in support_cache:
                support_cache[state.room] = room_is_supported(simulator, state.room, state)
            if not support_cache[state.room]:
                return
        support = ground_support_key(simulator, state)
        if support is None:
            return
        band, relative_x = _ground_arrival_band(simulator, state, support)
        bucket = ground_records.setdefault(support, {})
        # Parent-node reference plus a macro prefix avoids reconstructing a
        # potentially long route on every physics frame. Only retained records
        # are materialized once, after the bounded search finishes.
        record = (state, parent, mask, count, relative_x)
        _retain_ground_record(bucket, band, record, max_ground_arrivals_per_support)

    while frontier and not reached and expanded < max_expansions:
        if preview is not None:
            preview.report(preview_paths,expanded,simulated)
        if time.monotonic() >= deadline:
            reason = "time_budget"
            break
        _, index = heapq.heappop(frontier)
        state = nodes[index][0]
        elapsed = state.frame-start.frame
        if elapsed > best_cost.get(state_key(state),math.inf) or elapsed >= max_frames:
            continue
        expanded += 1
        for mask, duration in macros(state, heuristic, variant,simulator):
            successor = state
            used = 0
            crossings=[]
            masks=mask if isinstance(mask,tuple) else [mask]*duration
            for input_mask in masks[:max_frames-elapsed]:
                before=successor
                successor = simulator.step(successor,input_mask)
                if successor.room!=before.room:
                    crossings.extend((before,successor))
                simulated += 1
                used += 1
                if ground_records is not None:
                    collect_ground(successor, index, mask, used)
                if successor.dead or goal.achieved(successor) or first_hit_rejected(successor):
                    break
                if not allow_unsupported and successor.room!=before.room:
                    if successor.room not in support_cache:
                        support_cache[successor.room]=room_is_supported(simulator,successor.room,successor)
                    if not support_cache[successor.room]:
                        # Even an unsuccessful lane crossing must stop here;
                        # a terminal certificate never licenses another input
                        # inside the otherwise unsupported destination.
                        break
            if successor.dead or not used or first_hit_rejected(successor):
                continue
            if not allow_unsupported:
                if successor.room not in support_cache:
                    support_cache[successor.room] = room_is_supported(simulator,successor.room,successor)
                supported = support_cache[successor.room]
                if not supported and not terminal_goal_crossing_allowed(simulator,goal,before,successor):
                    continue
            cost = successor.frame-start.frame
            key = state_key(successor)
            if cost >= best_cost.get(key,math.inf):
                continue
            if continuation is not None and not continuation.claim():
                # Keep the unexpanded parent available after another search
                # releases memory; a memory cap never proves this edge absent.
                heapq.heappush(frontier,(heuristic(state),index))
                memory_limited = True
                reason = "continuation_memory_budget"
                break
            best_cost[key] = cost
            child_index = len(nodes)
            nodes.append((successor,index,mask,used,tuple(crossings)))
            h = heuristic(successor)
            if goal.achieved(successor):
                best_index, reached, reason = child_index, True, "goal"
                break
            # Near a pickup, explicitly evaluate a coast-and-land edge. Without
            # this macro, fine velocity variants can exhaust the budget before
            # the search ever expands the quiet landing/collection sequence.
            # Every frame still goes through exactly the same offline physics.
            if goal.kind=="berry" and goal.id in successor.following:
                settled=successor
                wait_crossings=[]
                for wait in range(1,min(180,max_frames-cost)+1):
                    before=settled
                    settled=simulator.step(settled,0)
                    if settled.room!=before.room:
                        wait_crossings.extend((before,settled))
                    simulated+=1
                    if ground_records is not None:
                        collect_ground(settled, child_index, 0, wait)
                    if settled.dead:
                        break
                    if not allow_unsupported:
                        if settled.room not in support_cache:
                            support_cache[settled.room]=room_is_supported(simulator,settled.room,settled)
                        if not support_cache[settled.room]:
                            break
                    if goal.achieved(settled):
                        if continuation is not None and not continuation.claim():
                            heapq.heappush(frontier,(heuristic(state),index))
                            memory_limited = True
                            reason = "continuation_memory_budget"
                            break
                        best_index=len(nodes)
                        nodes.append((settled,child_index,0,wait,tuple(wait_crossings)))
                        reached,reason=True,"goal"
                        break
                if reached:
                    break
            if h < best_h:
                best_index, best_h = child_index,h
            cost_weight=0.03 if variant%2==0 else 0.18
            heapq.heappush(frontier,(cost*cost_weight+h*(1.4+(variant%6)*0.15),child_index))
        if memory_limited:
            break
        if len(frontier)>40000:
            frontier = heapq.nsmallest(20000,frontier)
            heapq.heapify(frontier)
    if not frontier and not reached:
        reason = "frontier_exhausted"
    runs = []
    node = best_index
    while nodes[node][1]>=0:
        _,parent,mask,count = nodes[node][:4]
        runs.append(list(mask[:count]) if isinstance(mask,tuple) else [mask]*count)
        node = parent
    actions = [a for run in reversed(runs) for a in run]
    if continuation is not None:
        continuation.session = (heuristic,nodes,frontier,best_cost,best_index,best_h,reached)
    elif len(nodes)<50000:
        _SEARCH_SESSIONS[archive_key]=(heuristic,nodes,frontier,best_cost,best_index,best_h,reached)
        while len(_SEARCH_SESSIONS)>24 or sum(len(s[1]) for s in _SEARCH_SESSIONS.values())>80000:
            _SEARCH_SESSIONS.popitem(last=False)
    result_state=replace(nodes[best_index][0],frame=nodes[best_index][0].frame+original_frame)
    if ground_records is not None:
        for support, bucket in ground_records.items():
            for state, parent, mask, count, _ in bucket.values():
                runs = [tuple(mask[:count]) if isinstance(mask, tuple) else (mask,)*count]
                while nodes[parent][1] >= 0:
                    _, ancestor, edge, length = nodes[parent][:4]
                    runs.append(tuple(edge[:length]) if isinstance(edge, tuple) else (edge,)*length)
                    parent = ancestor
                inputs = tuple(action for run in reversed(runs) for action in run)
                ground_arrivals.append(GroundArrival(inputs,
                    replace(state, frame=state.frame+original_frame), support))
    if preview is not None:
        preview.report(preview_paths,expanded,simulated,remember=True)
    return SearchResult(actions,result_state,reached,expanded,simulated,reason)


def coverage_goals(room):
    """Sample supports, excluding obvious spike tops; these are search goals.

    Generating a goal never records an explored cell. Only an observed player
    trace may do that. Jump-throughs matter as much as solid tile supports.
    """
    goals = []
    points = set()
    for ty in range(2,room.height//8):
        run=[]
        for tx in range(room.width//8+1):
            exposed=(tx<room.width//8 and room.tile_at(tx,ty)
                     and not room.tile_at(tx,ty-1) and not room.tile_at(tx,ty-2))
            if exposed:
                run.append(tx)
            elif run:
                # Every short ledge gets its centre; a fixed global 32px grid
                # can miss a one-tile support that is essential for a climb.
                points.add((room.x+(run[0]+run[-1]+1)*4,room.y+ty*8))
                for column in run[1:-1:4]:
                    points.add((room.x+column*8+4,room.y+ty*8))
                run=[]
    for entity in room.entities:
        if entity.kind == "jumpThru":
            points.add((entity.x+entity.width/2,entity.y))
    for x,y in sorted(points):
        if any(e.kind == "spikesUp" and abs(e.y-y) < 4
               and e.x-5 < x < e.x+max(8,e.width)+5 for e in room.entities):
            continue
        goals.append(Goal("surface",room.name,x,y))
    return goals


def plan_campaign(simulator, map_data, start_room, *, seconds=30, expansions=5000,
                  max_goals=32, variant=0, allow_unsupported=False,
                  target: Goal | None=None, prefix: list[int] | None=None):
    simulator.map_data = map_data
    started = time.monotonic()
    deadline = started+seconds
    state = simulator.initial_state(start_room)
    actions = list(prefix or [])
    rooms, cells = {state.room}, set()
    for mask in actions:
        state = simulator.step(state,mask)
        if state.dead:
            raise ValueError("Resume route does not survive offline replay")
        rooms.add(state.room)
        cells.add((state.room,int(state.x//16),int(state.y//16)))
    goals = ([target] if target else
             [Goal("berry",r.name,e.x,e.y,berry_id(r.name,e)) for r in map_data.rooms.values()
              for e in ordinary_berries(r)] +
             [Goal("room",r.name,*r.spawns[0]) for r in map_data.rooms.values() if r.spawns] +
             [g for r in map_data.rooms.values() if r.spawns for g in coverage_goals(r)])
    completed = set()
    attempted_at = {}
    blocked = set()
    supported = {name:room_is_supported(simulator,name) for name in map_data.rooms}
    skipped, legs = [], []
    total_expansions = total_frames = 0
    while time.monotonic()<deadline and len(legs)<max_goals:
        candidates = []
        for goal in goals:
            if goal.key in completed or goal.key in blocked or attempted_at.get(goal.key)==len(actions):
                continue
            if goal.achieved(state) or (goal.kind=="room" and goal.room in rooms):
                completed.add(goal.key)
                continue
            if not allow_unsupported and not supported[goal.room]:
                blocked.add(goal.key)
                skipped.append({"goal":goal.key,"reason":"unsupported_mechanics"})
                continue
            distance = math.hypot(state.x-goal.x,state.y-goal.y)
            category = {"berry":0,"room":1,"explore":2,"surface":2}[goal.kind]
            # Prioritize all berries in this room, then adjacent rooms, then its surfaces.
            adjacent = {r.name for r in map_data.neighbors(state.room)}
            priority = (0 if goal.room==state.room and category==0 else
                        1 if category==1 and goal.room in adjacent else
                        2 if goal.room==state.room and category==2 else
                        3 if category==0 else 4)
            candidates.append((priority,distance,goal.key,goal))
        if not candidates:
            break
        _,_,_,goal = min(candidates)
        # A difficult target cannot consume the entire chapter budget.
        leg_deadline = deadline if target else min(deadline,time.monotonic()+max(1.0,seconds/6))
        from .strategies import search_goal
        result = search_goal(simulator,state,goal,deadline=leg_deadline,
                            max_expansions=expansions,variant=variant,
                            allow_unsupported=allow_unsupported)
        total_expansions += result.expansions
        total_frames += result.simulated_frames
        attempted_at[goal.key]=len(actions)
        # Preserve the connected route, but don't commit failed detours.
        if result.reached:
            completed.add(goal.key)
            for mask in result.actions:
                state = simulator.step(state,mask)
                rooms.add(state.room)
                cells.add((state.room,int(state.x//16),int(state.y//16)))
            actions.extend(result.actions)
        elif target and result.actions:
            # Explicit target mode may deliver the best partial prefix, labeled partial.
            state = result.state
            actions.extend(result.actions)
        legs.append({"goal":goal.key,"kind":goal.kind,"reached":result.reached,
                     "reason":result.reason,"expansions":result.expansions,
                     "frames":len(result.actions) if result.reached or target else 0})
        if target:
            break
    return {"actions":actions,"state":state,"legs":legs,"skipped":skipped,
            "expansions":total_expansions,"simulated_frames":total_frames,
            "wall_seconds":round(time.monotonic()-started,3),"variant":variant,
            "pending_goals":[g.key for g in goals if g.key not in completed and not g.achieved(state)
                             and not(g.kind=="room" and g.room in rooms)],
            "scope":"offline_prediction_only"}
