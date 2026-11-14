"""Search from physically reached ground intervals, with exact public endpoints.

Intervals are private guidance, never alternate player snapshots or success
conditions. Each start seed includes its complete walk/jump prefix. A private
landing is followed by a continuous, no-dash walk/jump suffix into the original
public target. Timers, moving solids, resources and ordered segment limits are
therefore carried through all three pieces.
"""
from dataclasses import dataclass, field, replace
import heapq
import math
import time

from .assist_routes import advance_route_progress, route_complete, start_route_progress
from .corridor_planning import _ConstrainedSimulator, _StageGoal
from .mechanism_moves import active_dash_block_edges
from .mechanism_routes import dash_break_goals
from .physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP, _spike_rect
from .planner import (Goal, ground_support_key, search_leg, room_is_supported,
                      terminal_goal_crossing_allowed)
from .room_planning import action_fingerprint
from .search_progress import remember_segment
from .step_cache import ExactStepCache


@dataclass(frozen=True)
class GroundInterval:
    room: str
    left: float
    right: float
    y: float
    identity: str = ""
    anchor_x: float = 0.
    anchor_y: float = 0.

    @property
    def key(self):
        return f"ground:{self.room}:{self.identity}:{self.left:g}:{self.right:g}:{self.y:g}"

    def bounds(self, simulator, state):
        if state.room != self.room:
            return None
        if not self.identity:
            return self.left, self.right, self.y
        for identity, _, rect in simulator._mechanism_rects(self.room, state):
            if identity == self.identity:
                dx, dy = rect[0]-self.anchor_x, rect[1]-self.anchor_y
                return self.left+dx, self.right+dx, self.y+dy
        return None


@dataclass(frozen=True)
class LandingRegionGoal(Goal):
    region: GroundInterval | None = None
    simulator: object = field(default=None, compare=False, hash=False, repr=False)

    def waypoint(self, state):
        bounds = self.region.bounds(self.simulator, state)
        if bounds is None:
            return self.x, self.y
        left, right, y = bounds
        return min(right, max(left, state.x)), y

    def achieved(self, state):
        if state.dead or not state.on_ground or state.vy < 0:
            return False
        bounds = self.region.bounds(self.simulator, state)
        if bounds is None:
            return False
        left, right, y = bounds
        return left <= state.x <= right and abs(state.y-y) < 1.1


@dataclass(frozen=True)
class _ExitApproachGoal(Goal):
    public_goal: object = None

    def __getattr__(self, name):
        return getattr(self.public_goal, name)

    def waypoint(self, state):
        return (self.x, self.y) if state.room == self.room else self.public_goal.waypoint(state)

    def achieved(self, state):
        return self.public_goal.achieved(state)


def _exit_approaches(goal, state):
    if not getattr(goal, "transition_to", "") or getattr(goal, "source_room", "") != state.room:
        return (goal,)
    members = getattr(goal, "region_members", lambda: ())()
    if not members:
        return (goal,)
    choices = []
    for member in members:
        left, top, right, bottom = member["arrival_bounds"]
        x, y = member["x"]+4, member["y"]+9.5
        if goal.edge in ("top", "bottom"):
            x = (left+right)/2
        else:
            y = (top+bottom)/2+5.5
        choices.append(_ExitApproachGoal(goal.kind, state.room, x, y,
            id=f"{goal.key}:approach:{x:g}:{y:g}", public_goal=goal))
    choices.sort(key=lambda item: math.hypot(state.x-item.x, state.y-item.y))
    # Nearest plus both ends give source-side obstacle detours a chance while
    # success still accepts the entire certified public exit region.
    extremes = sorted(choices, key=lambda item:(item.x, item.y))
    selected = [choices[0], extremes[0], extremes[-1]]
    return tuple(dict.fromkeys(selected))


def _subtract(intervals, left, right):
    """Remove an open blocked interval; exact contact at an edge remains legal."""
    result = []
    for a, b in intervals:
        if right <= a or left >= b:
            result.append((a, b))
        else:
            if left >= a:
                result.append((a, min(b, left)))
            if right <= b:
                result.append((max(a, right), b))
    return result


def ground_intervals(simulator, state):
    """Continuous safe support geometry at this *actual* mechanism snapshot.

    Work scales with tile spans and obstacle edges, not every player pixel.
    The result proposes destinations only; every accepted arrival is simulated.
    Moving supports retain identity so later goals follow their real position.
    """
    room = simulator.map.rooms.get(state.room)
    if room is None:
        return ()
    dynamic = tuple(simulator._mechanism_rects(state.room, state))
    solids = tuple(simulator._static_rects[state.room]) + tuple(row[2] for row in dynamic)
    surfaces = {}
    # Merge neighboring exposed tiles before eroding by the player half-width.
    for left, top, right, _ in simulator._static_rects[state.room]:
        surfaces.setdefault((top, "", 0., 0.), []).append((left, right))
    for left, top, right, _ in simulator._platforms[state.room]:
        surfaces.setdefault((top, "", 0., 0.), []).append((left, right))
    for identity, _, (left, top, right, _) in dynamic:
        surfaces.setdefault((top, identity, left, top), []).append((left, right))
    hazards = []
    records = {record[0]: record for record in state.mechanisms}
    for entity in simulator._hazards[state.room]:
        x, y = entity.x, entity.y
        mover = simulator._spike_attachments[state.room].get((entity.kind, entity.id))
        if mover is not None:
            identity = f"{state.room}:{mover.id}"
            record = records.get(identity)
            if identity in state.broken or record and mover.kind == "fallingBlock" and record[1] == 4:
                continue
            if record:
                x += round(record[3])-mover.x
                y += round(record[4])-mover.y
        rect = ((x-6, y-6, x+6, y+6) if entity.kind == "spinner"
                else _spike_rect(entity, x, y))
        hazards.append(rect)
    result = []
    for (y, identity, anchor_x, anchor_y), spans in surfaces.items():
        merged = []
        for left, right in sorted(spans):
            if merged and left <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(right, merged[-1][1])
            else:
                merged.append((left, right))
        intervals = [(max(room.x+4, left+4), min(room.x+room.width-4, right-4))
                     for left, right in merged if right-left >= 8]
        for left, top, right, bottom in solids:
            if top < y and bottom > y-11:
                intervals = _subtract(intervals, left-4, right+4)
        for left, top, right, bottom in hazards:
            if top < y-2 and bottom > y-11:
                intervals = _subtract(intervals, left-4-.01, right+4+.01)
        for left, right in intervals:
            if left <= right:
                result.append(GroundInterval(state.room, left, right, y,
                                             identity, anchor_x, anchor_y))
    return tuple(sorted(result, key=lambda r: (r.y, r.left, r.right, r.identity)))


def target_ground_intervals(simulator, state, goal):
    """Candidate floors from which an ordinary final walk/jump might hit goal.

    A floor may extend far from the selected tile; the main search may enter at
    any safe point on it. Geometry does not certify the final jump, and an
    airborne exit remains governed solely by its public transition predicate.
    """
    if (goal.room != state.room or getattr(goal, "transition_to", "")
            or goal.kind not in ("target", "surface")):
        return ()
    regions = []
    for region in ground_intervals(simulator, state):
        # Normal jump apex is roughly forty pixels. Small geometry margins
        # keep useful proposals; actual no-dash suffix replay is decisive.
        if -4 <= region.y-goal.y <= 52 and region.left-56 <= goal.x <= region.right+56:
            regions.append(region)
    return tuple(sorted(regions, key=lambda r: (max(r.left-goal.x, goal.x-r.right, 0),
                                                 abs(r.y-goal.y), -r.right+r.left))[:8])


@dataclass
class _Arrival:
    state: object
    actions: list
    phase: object
    stages: tuple = ()
    dashes: int = 0


@dataclass
class _Expansion:
    node: _Arrival
    iterator: object = None


def _pop_stage_support(queue, turn):
    """Prefer completed public points, keeping one turn for earlier arrivals.

    The oldest other-stage support gets every fourth turn. Its existing FIFO
    still rotates full-state arrivals and suspended work on that support.
    """
    if turn % 4 == 3:
        deepest = queue[0][0]
        alternatives = [(row[1], index) for index, row in enumerate(queue)
                        if row[0] != deepest]
        if alternatives:
            _, index = min(alternatives)
            selected = queue[index]
            queue[index] = queue[-1]
            queue.pop()
            heapq.heapify(queue)
            return selected
    return heapq.heappop(queue)


def _diverse_arrivals(arrivals, goal, limit=12):
    """Keep support ends, momentum, opened blocks and mechanism clocks."""
    if len(arrivals) <= limit:
        return arrivals
    def score(node):
        point = getattr(goal, "waypoint", lambda state: (goal.x, goal.y))(node.state)
        return math.hypot(node.state.x-point[0], node.state.y-point[1]) + len(node.actions)*.025
    by_band = {}
    for node in sorted(arrivals, key=score):
        s = node.state
        band = (node.phase.index, s.room, round(s.y/8), int(s.x//24),
                (s.vx > 5)-(s.vx < -5), s.dashes, s.broken,
                tuple((row[0], row[1], row[2]//12) for row in s.mechanisms))
        by_band.setdefault(band, node)
    rows = list(by_band.values())
    # Extreme real arrivals reserve a launch on either end of long platforms;
    # goal-distance alone would remove precisely the run-up/backtracking seed.
    retained = [arrivals[0]]
    worlds = {arrivals[0].state.broken}
    # Opening a block changes reachable geometry. A crowd of nearby ordinary
    # arrivals must not evict the only genuine successor in the opened world.
    for node in rows:
        if node.state.broken not in worlds:
            worlds.add(node.state.broken)
            retained.append(node)
    for node in (min(rows, key=lambda n:n.state.x), max(rows, key=lambda n:n.state.x)):
        if all(node is not old for old in retained):
            retained.append(node)
    for node in rows:
        if all(node is not old for old in retained):
            retained.append(node)
        if len(retained) >= limit:
            break
    return retained[:limit]


def _nearby_breaks(simulator, seeds, limit=2):
    """Select nearby faces from complete real arrivals, once per block.

    This is local action discovery, not a remote block-approach search. The
    guide poses only rank candidates; the returned seed is never relocated.
    """
    candidates = {}
    for seed in seeds:
        state = seed.state
        for target in dash_break_goals(simulator, state):
            left, top, right, bottom = target.rect
            if target.dash_dx:
                if (target.dash_dx > 0 and state.x > left-3
                        or target.dash_dx < 0 and state.x < right+3):
                    continue
                aligned = top < state.y and state.y-11 < bottom
            else:
                if (target.dash_dy > 0 and state.y > top
                        or target.dash_dy < 0 and state.y-11 < bottom):
                    continue
                aligned = state.x+4 > left and state.x-4 < right
            distance = math.hypot(state.x-target.x, state.y-target.y)
            if distance > 64:
                continue
            rank = (not aligned, distance, seed.dashes, len(seed.actions))
            previous = candidates.get(target.identity)
            if previous is None or rank < previous[0]:
                candidates[target.identity] = (rank, seed, target)
    return [(seed, target) for _, seed, target in
            sorted(candidates.values(), key=lambda row: row[0])[:limit]]


def _support_arrivals(simulator, arrivals, goal, limit=12):
    """Reserve representatives on every discovered support, even far from goal.

    The limit applies within a support, never across the room. The selected
    nodes keep their complete physical state and actual input prefix.
    """
    groups = {}
    for node in arrivals:
        key = (node.phase, ground_support_key(simulator, node.state))
        groups.setdefault(key, []).append(node)
    return [node for rows in groups.values()
            for node in _diverse_arrivals(rows, goal, limit)]


def _world_state(state):
    """Changes that can make revisiting the same support useful."""
    return (state.berries, state.following, state.fled, state.broken,
            state.refill_timers, state.mechanisms, state.cassette_clock,
            state.cassette_blocks, state.cassette_stopped)


class _WalkJumpSimulator(_ConstrainedSimulator):
    """The final regional connector consumes only ordinary directional jumps."""
    def step(self, before, mask):
        if mask & (DASH | GRAB | UP | DOWN):
            return replace(before, dead=True, reason="regional_walk_jump_only")
        return super().step(before, mask)


def search_regional_routes(simulator, start, route, goals, *, deadline,
                           max_expansions, max_frames, variant=0, forbidden=()):
    """Bounded multi-source, ground-region search with ordered exact replay.

    The seven-item return follows corridor/mechanism planners. Failure carries
    only a real best prefix and is never an executable success. At most twelve
    local seeds per support and eight target intervals guide each expansion.
    Failed probes also contribute their physically reached ground frontiers.
    """
    simulator = ExactStepCache(simulator)
    phase = start_route_progress(route, start)
    root = _Arrival(start, [], phase)
    stats = dict(expansions=0, simulated_frames=0)
    details = dict(regional_guidance=True, start_seeds=[], target_regions=[],
                   attempts=[], stages=[], backtracks=0, excluded_candidates=0,
                   active_breaks=dict(attempted=0, accepted=0, expansions=0,
                                      max_per_node=2, max_probe_expansions=32))
    ground_stats = details["ground_frontier"] = dict(
        accepted=0, expanded=0, duplicates=0, limited=0, supports=[],
        max_pending_per_support=0, resumed=0)
    blacklist = set(forbidden)
    best = root
    seen = {}
    pending, scheduled, known_supports = {}, set(), set()
    queue = []
    serial = 0
    selection_turn = 0
    if not goals or len(goals) != len(route["waypoints"]) or phase.violation:
        details["reason"] = "invalid_public_goals"
        return [], start, False, 0, 0, phase, details

    def available():
        return time.monotonic() < deadline and stats["expansions"] < max_expansions

    def finish(node, reached=False):
        details["stages"] = list(node.stages)
        details["step_cache"] = simulator.cache_info()
        ground_stats["pending"] = sum(len(rows) for rows in pending.values())
        return (node.actions, node.state, reached, stats["expansions"],
                stats["simulated_frames"], node.phase, details)

    def support_key(node):
        support = ground_support_key(simulator, node.state)
        # An airborne public waypoint is still a legitimate next segment root.
        return (node.phase.index, support or ("air", node.state.room,
                                              round(node.state.x/24), round(node.state.y/24)))

    def activate(key):
        nonlocal serial
        rows = pending.get(key)
        if key in scheduled or not rows:
            return
        scheduled.add(key)
        serial += 1
        heapq.heappush(queue, (-key[0], serial, key))

    def retain_pending(support):
        rows = pending[support]
        # Paused expansions and unstarted arrivals share the same allowance.
        # Retaining the origin node also retains its suspended search position.
        if len(rows) > 12:
            selected = _diverse_arrivals([work.node for work in rows],
                                         goals[support[0]], 12)
            selected_ids = {id(node) for node in selected}
            pending[support] = [work for work in rows if id(work.node) in selected_ids]
            ground_stats["limited"] += 1
        ground_stats["max_pending_per_support"] = max(
            ground_stats["max_pending_per_support"], len(pending[support]))

    def replay(node, actions, label, *, no_dash=False, certify=False):
        current, progress, used, states, dashes = node.state, node.phase, [], [node.state], 0
        for mask in actions[:max(0, max_frames-len(node.actions))]:
            if not certify and time.monotonic() >= deadline:
                return None
            before = current
            current = simulator.step(before, mask)
            started_dash = bool(getattr(simulator, "last_step_started_dash", False))
            stats["simulated_frames"] += 1
            if no_dash and (mask & DASH or started_dash):
                return None
            dashes += started_dash
            progress = advance_route_progress(route, progress, before, current, mask)
            if current.dead or progress.violation:
                return None
            used.append(mask)
            states.append(current)
            if current.room != before.room and not room_is_supported(simulator, current.room, current):
                if not terminal_goal_crossing_allowed(simulator, goals[node.phase.index], before, current):
                    return None
            if progress.index > node.phase.index:
                break
        if not used:
            return None
        remember_segment(node.state, current, states)
        return _Arrival(current, node.actions+used, progress, node.stages+(label,), node.dashes+dashes)

    def accept(node):
        nonlocal best, serial
        if node is None:
            return None
        if route_complete(route, node.phase):
            if (action_fingerprint(node.actions) in blacklist
                    or getattr(goals[-1], "first_hit_rejected", lambda state: False)(node.state)):
                details["excluded_candidates"] += 1
                return None
            return node
        if (node.phase.index, -node.dashes, -len(node.actions)) > (best.phase.index, -best.dashes, -len(best.actions)):
            best = node
        if len(node.actions) >= max_frames:
            return None
        # Absolute frame is bookkeeping; mechanism/cassette clocks and every
        # movement field remain in the key. Never normalize the actual node.
        key = (replace(node.state, frame=0, reason=""), node.phase)
        costs = seen.setdefault(key, [])
        cost = (node.dashes, len(node.actions))
        if any(dashes <= cost[0] and frames <= cost[1] for dashes, frames in costs):
            ground_stats["duplicates"] += 1
            return None
        costs[:] = [old for old in costs if not (cost[0] <= old[0] and cost[1] <= old[1])]
        costs.append(cost)
        support = support_key(node)
        if support not in known_supports:
            known_supports.add(support)
            ground_stats["supports"].append(dict(public_index=node.phase.index,
                room=node.state.room, x=node.state.x, y=node.state.y,
                key=repr(support[1])))
        rows = pending.setdefault(support, [])
        rows.append(_Expansion(node))
        # Bound variants *within* a floor, never drop a different floor because
        # it is farther from the final click. Full states survive selection.
        retain_pending(support)
        ground_stats["accepted"] += 1
        activate(support)
        return None

    def local_arrivals(node, goal, *, suffix=False, limit=12):
        """Actual no-dash walk/jump prefixes; no fabricated landing snapshots."""
        if not available() or len(node.actions) >= max_frames:
            return [node]
        arrivals = [node]
        hits = []
        walks = [node]
        aim_x, _ = getattr(goal, "waypoint", lambda state: (goal.x, goal.y))(node.state)
        direction = RIGHT if aim_x >= node.state.x else LEFT
        frames = max_frames-len(node.actions)
        for move in (direction, LEFT if direction == RIGHT else RIGHT):
            if not available():
                break
            stats["expansions"] += 1
            current, progress = node.state, node.phase
            used, states = [], [current]
            for tick in range(frames):
                if time.monotonic() >= deadline:
                    break
                before, before_progress = current, progress
                current = simulator.step(current, move)
                stats["simulated_frames"] += 1
                progress = advance_route_progress(route, progress, before, current, move)
                if (current.dead or progress.violation or current.room != node.state.room
                        or getattr(simulator, "last_step_started_dash", False)):
                    # Crossings are handled by the exact-goal searches, which
                    # also enforce terminal-entry certificates.
                    break
                used.append(move)
                states.append(current)
                if progress.index > node.phase.index:
                    child = _Arrival(current, node.actions+used[:], progress,
                                     node.stages+("ground_exact_suffix" if suffix else "ground_start_walk",), node.dashes)
                    remember_segment(node.state, current, states)
                    hits.append(child)
                    break
                if not current.on_ground or current.vy < 0:
                    if before.on_ground and before.vy >= 0 and len(used) > 1:
                        child = _Arrival(before, node.actions+used[:-1], before_progress,
                            node.stages+("ground_exact_walk" if suffix else "ground_start_walk",), node.dashes)
                        remember_segment(node.state, before, states[:-1])
                        arrivals.append(child)
                        walks.append(child)
                    break
                if tick % 12 == 5 or current.x == before.x or tick == frames-1:
                    child = _Arrival(current, node.actions+used[:], progress,
                                     node.stages+("ground_exact_walk" if suffix else "ground_start_walk",), node.dashes)
                    remember_segment(node.state, current, states)
                    arrivals.append(child)
                    walks.append(child)
                if tick > 8 and current.x == before.x and abs(current.vx) < .01:
                    break
        # Include launch positions at both support ends and near the target.
        launchers = _diverse_arrivals(walks, goal, 6)
        for launch in launchers:
            for move in (direction, LEFT if direction == RIGHT else RIGHT, 0):
                for hold in (5, 16):
                    if not available():
                        return hits + _support_arrivals(simulator, arrivals, goal, limit)
                    stats["expansions"] += 1
                    masks = [0] + [move|JUMP]*hold + [move]*64
                    # Releasing JUMP before takeoff is intentional: seeds retain
                    # their actual held-button state from the incoming route.
                    current, progress, used, states = launch.state, launch.phase, [], [launch.state]
                    airborne = False
                    for mask in masks[:max_frames-len(launch.actions)]:
                        if time.monotonic() >= deadline:
                            break
                        before = current
                        current = simulator.step(before, mask)
                        stats["simulated_frames"] += 1
                        progress = advance_route_progress(route, progress, before, current, mask)
                        if (current.dead or progress.violation or current.room != launch.state.room
                                or getattr(simulator, "last_step_started_dash", False)):
                            break
                        used.append(mask)
                        states.append(current)
                        airborne |= not current.on_ground or current.vy < 0
                        child = None
                        if progress.index > node.phase.index:
                            child = _Arrival(current, launch.actions+used[:], progress,
                                launch.stages+("ground_exact_suffix" if suffix else "ground_start_jump",), launch.dashes)
                            hits.append(child)
                        elif airborne and current.on_ground and current.vy >= 0:
                            child = _Arrival(current, launch.actions+used[:], progress,
                                launch.stages+("ground_suffix_setup" if suffix else "ground_start_jump",), launch.dashes)
                            arrivals.append(child)
                        if child:
                            remember_segment(launch.state, current, states)
                            break
        return hits + _support_arrivals(simulator, arrivals, goal, limit)

    def probe(node, target, *, no_dash, budget, label, ordinary=False):
        if not available() or budget <= 0 or len(node.actions) >= max_frames:
            return None, []
        wrapper = _WalkJumpSimulator if ordinary else _ConstrainedSimulator
        constrained = wrapper(simulator, route, phase=node.phase, no_dash=no_dash)
        public = goals[node.phase.index]
        if target is public:
            target = _exit_approaches(public, node.state)[0]
        stage = (target if target is public else _StageGoal(target.kind, target.room,
                 target.x, target.y, id=target.key, portal=target, public_goal=public))
        landings = []
        result = search_leg(constrained, node.state, stage, deadline=min(deadline, time.monotonic()+3.),
            max_expansions=min(budget, max_expansions-stats["expansions"]),
            max_frames=max_frames-len(node.actions), variant=variant,
            ground_arrivals=landings)
        stats["expansions"] += result.expansions
        stats["simulated_frames"] += result.simulated_frames
        if len(details["attempts"]) < 128:
            details["attempts"].append(dict(public_index=node.phase.index, kind=label,
                no_dash=no_dash, reached=result.reached, expansions=result.expansions,
                frames=len(result.actions), start_x=node.state.x, start_y=node.state.y))
        child = replay(node, result.actions, label, no_dash=no_dash) if result.reached else None
        continuations = []
        for landing in landings:
            # A return to the same floor can carry different launch momentum,
            # lift/force timers or input edges. Only exact-state dominance may
            # discard it before the bounded per-support representative queue.
            arrival = replay(node, landing.actions, "ground_frontier_landing", no_dash=no_dash)
            if arrival is not None:
                continuations.append(arrival)
        return child, continuations

    def useful_ground(node, state):
        """The same unchanged floor already has local walk/jump launch seeds."""
        support = ground_support_key(simulator, state)
        return support is not None and (
            support != ground_support_key(simulator, node.state)
            or _world_state(state) != _world_state(node.state)
            or state.dashes > node.state.dashes)

    def expand(node):
        goal = goals[node.phase.index]
        segment = route["segments"][node.phase.index]
        seeds = local_arrivals(node, goal)
        ongoing = []
        for seed in seeds:
            if seed.phase.index > node.phase.index:
                yield seed
            else:
                ongoing.append(seed)
                if useful_ground(node, seed.state):
                    yield seed
        seeds = _diverse_arrivals(ongoing, goal, 12)
        if len(details["start_seeds"]) < 80:
            details["start_seeds"].extend(dict(public_index=node.phase.index, x=s.state.x,
                y=s.state.y, vx=s.state.vx, frames=len(s.actions), dashes=s.state.dashes)
                for s in seeds)
        if not seeds:
            return
        # A cheap no-dash exact continuation also handles goals above the
        # current floor without demanding a redundant private landing.
        # One cheap exact probe per continuation. Other physical launches keep
        # their region/exit attempts below; repeating this opening probe for
        # every local seed would starve the newly shared ground frontier.
        for seed in seeds[:1]:
            child, landings = probe(seed, goal, no_dash=True, budget=100, label="ground_exact_no_dash")
            if child and child.phase.index > node.phase.index:
                yield child
            yield from landings
            yield None
        # Actual nearby arrivals can deliberately open a new path. Keep this
        # small and local; distant block approaches belong to the mechanism
        # planner. A rebound is a valid continuation even before it lands.
        if not segment.get("no_dash"):
            for seed, target in _nearby_breaks(simulator, seeds):
                if not available() or len(seed.actions) >= max_frames:
                    break
                constrained = _ConstrainedSimulator(simulator, route, phase=seed.phase)
                stage = _StageGoal(target.kind, target.room, target.x, target.y,
                    id=target.key, portal=target, public_goal=goal)
                before = stats["expansions"]
                rows = active_dash_block_edges(constrained, seed.state, stage,
                    deadline=min(deadline, time.monotonic()+.2),
                    max_frames=min(180, max_frames-len(seed.actions)), stats=stats,
                    max_expansions=min(max_expansions, before+32), max_edges=2)
                details["active_breaks"]["attempted"] += 1
                details["active_breaks"]["expansions"] += stats["expansions"]-before
                children = []
                for actions, _, label in rows:
                    # Certify only already-computed bounded inputs after the
                    # search slice ends. No successor or broken flag is trusted.
                    child = replay(seed, actions, "ground_"+label, certify=True)
                    if child is not None and (child.phase.index > seed.phase.index
                            or target.identity in child.state.broken-seed.state.broken):
                        children.append(child)
                details["active_breaks"]["accepted"] += len(children)
                if len(details["attempts"]) < 128:
                    details["attempts"].append(dict(public_index=seed.phase.index,
                        kind="ground_active_dash_block", identity=target.identity,
                        start_x=seed.state.x, start_y=seed.state.y,
                        expansions=stats["expansions"]-before,
                        reached=any(target.identity in child.state.broken for child in children),
                        public_hit=any(child.phase.index > seed.phase.index for child in children),
                        frames=max((len(child.actions)-len(seed.actions) for child in children), default=0)))
                # A completed public hit must survive the final search slice;
                # private destruction alone never completes the public route.
                yield from sorted(children, key=lambda child: -child.phase.index)
                yield None
        regions = (() if segment.get("no_ground") else target_ground_intervals(simulator, node.state, goal))
        if len(details["target_regions"]) < 64:
            details["target_regions"].extend(dict(public_index=node.phase.index,
                room=r.room, left=r.left, right=r.right, y=r.y, identity=r.identity) for r in regions)
        # Region entry is intentionally nearer than the exact click. Once
        # there, a local ordinary walk/jump must hit the original public tile.
        region_limit = stats["expansions"] + int((max_expansions-stats["expansions"])*.7)
        modes = (True,) if segment.get("no_dash") else (True, False)
        # Preserve the ordinary-first preference without making every launch
        # finish every no-dash probe before any dash-capable continuation. Each
        # floor gets both modes from one real seed before trying another seed.
        for seed in seeds[:4]:
            for region_index, region in enumerate(regions):
                for mode_index, no_dash in enumerate(modes):
                    if not available() or stats["expansions"] >= region_limit:
                        break
                    # A small remaining allowance must not all disappear in
                    # the first ordinary probe. Reserve a bounded share for
                    # every region/mode still waiting on this same seed.
                    turns_left = (len(regions)-region_index)*len(modes)-mode_index
                    mode_limit = stats["expansions"] + max(1,
                        (region_limit-stats["expansions"])//turns_left)
                    bounds = region.bounds(simulator, seed.state)
                    if bounds is None:
                        continue
                    left, right, y = bounds
                    landing = LandingRegionGoal("surface", region.room,
                        min(right,max(left,seed.state.x)), y, id=region.key,
                        region=region, simulator=simulator)
                    if landing.achieved(seed.state):
                        arrivals = [seed] if no_dash else []
                    else:
                        arrivals = []
                        child, landings = probe(seed, landing, no_dash=no_dash, budget=min(240,mode_limit-stats["expansions"]),
                                      label="ground_region_landing")
                        if child:
                            arrivals.append(child)
                        yield from landings
                        yield None
                    for arrival in arrivals:
                        if arrival.phase.index > node.phase.index:
                            yield arrival
                            continue
                        if useful_ground(node, arrival.state):
                            yield arrival
                        suffixes = local_arrivals(arrival, goal, suffix=True, limit=4)
                        completed_suffix = False
                        for suffix in suffixes:
                            if suffix.phase.index > node.phase.index:
                                completed_suffix = True
                                yield suffix
                            elif useful_ground(arrival, suffix.state):
                                yield suffix
                        if not completed_suffix:
                            child, landings = probe(arrival, goal, no_dash=True, budget=max(0,min(180,mode_limit-stats["expansions"])),
                                          label="ground_exact_suffix", ordinary=True)
                            if child and child.phase.index > node.phase.index:
                                yield child
                            else:
                                details["backtracks"] += 1
                            yield from landings
                            yield None
        # Broader launch-position alternatives retain original airborne/end
        # semantics and permit a dash only when ordinary alternatives failed.
        for seed in seeds:
            for approach in _exit_approaches(goal, seed.state):
                child, landings = probe(seed, approach, no_dash=bool(segment.get("no_dash")),
                              budget=320, label="ground_seed_continuation")
                if child and child.phase.index > node.phase.index:
                    yield child
                yield from landings
                yield None
        details["backtracks"] += 1
    completed = accept(root)
    if completed:
        return finish(completed, True)
    while queue and available():
        _, _, support = _pop_stage_support(queue, selection_turn)
        selection_turn += 1
        scheduled.remove(support)
        work = pending[support].pop(0)
        if work.iterator is None:
            if work.node.actions:
                ground_stats["expanded"] += 1
            work.iterator = iter(expand(work.node))
        else:
            ground_stats["resumed"] += 1
        try:
            child = next(work.iterator)
        except StopIteration:
            activate(support)
            continue
        # One probe per turn. Distinct floors rotate fairly, including detours
        # away from the goal. Within a floor, rotate both suspended expansions
        # and new full-state arrivals so the oldest branch cannot monopolize it.
        pending[support].append(work)
        retain_pending(support)
        completed = accept(child)
        if completed:
            return finish(completed, True)
        activate(support)
    details["reason"] = ("time_budget" if time.monotonic() >= deadline else
                          "budget" if stats["expansions"] >= max_expansions else
                          "ground_frontier_limit" if ground_stats["limited"] else "regional_frontier_exhausted")
    return finish(best)
