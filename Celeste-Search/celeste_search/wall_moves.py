"""Geometry-guided wall relays made exclusively from simulated input frames."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from .physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP
from .planner import Goal
from .search_progress import remember_segment


@dataclass(frozen=True)
class WallGripGoal(Goal):
    rect: tuple = ()
    facing: int = 1
    identity: str = ""
    mechanism_kind: str = ""
    grip_offset: float = 12

    def bounds(self, state):
        left, top, right, bottom = self.rect
        if self.identity:
            if self.identity in state.broken:
                return None
            if self.mechanism_kind == "cassetteBlock":
                block = next((block for block in state.cassette_blocks
                              if block.id == self.identity), None)
                if block is None or not block.active or not block.collidable:
                    return None
                return block.rect
            record = next((row for row in state.mechanisms if row[0] == self.identity), None)
            if record is not None:
                if ((self.mechanism_kind == "fallingBlock" and record[1] == 4)
                        or (self.mechanism_kind == "crumbleBlock" and record[1] == 3)):
                    return None
                x, y = round(record[3]), round(record[4])
                return x, y, x+right-left, y+bottom-top
        return self.rect

    def waypoint(self, state):
        bounds = self.bounds(state)
        if bounds is None:
            return self.x, self.y
        left, top, right, _ = bounds
        return (left-4 if self.facing > 0 else right+4), min(self.y, top+self.grip_offset)

    def achieved(self, state):
        if state.dead or state.room != self.room or state.state != 1 or state.facing != self.facing:
            return False
        bounds = self.bounds(state)
        if bounds is None:
            return False
        left, top, right, bottom = bounds
        x = left-4 if self.facing > 0 else right+4
        return (abs(state.x-x) < .1 and state.y > top and state.y-11 < bottom
                and state.y <= self.y+4)


@dataclass(frozen=True)
class WallContactGoal(WallGripGoal):
    """A real side catch, without requiring an arbitrary guidance height."""

    def achieved(self, state):
        if (state.dead or state.room != self.room or state.state != 1
                or state.facing != self.facing or state.stamina <= 0):
            return False
        bounds = self.bounds(state)
        if bounds is None:
            return False
        left, top, right, bottom = bounds
        x = left-4 if self.facing > 0 else right+4
        return abs(state.x-x) < .1 and state.y > top and state.y-11 < bottom


def _static_faces(simulator, room):
    """Merge adjacent exposed tile sides, avoiding a goal for every tile."""
    columns = {}
    tiles = simulator._tiles[room.name]
    for tx, ty in tiles:
        for facing, neighbor in ((1, tx-1), (-1, tx+1)):
            if (neighbor, ty) not in tiles:
                edge = tx if facing > 0 else tx+1
                columns.setdefault((edge, facing), []).append(ty)
    for (edge, facing), rows in sorted(columns.items()):
        rows.sort()
        first = last = rows[0]
        for row in rows[1:]+[None]:
            if row == last+1:
                last = row
                continue
            x = room.x+edge*8
            if (last-first+1)*8 >= 8:
                # Only the selected side is used; width is irrelevant for an
                # exposed static face, but a real one-tile rectangle is handy.
                left = x if facing > 0 else x-8
                yield (left, room.y+first*8, left+8, room.y+(last+1)*8), facing
            first = last = row
    for rect in simulator._solids[room.name]:
        if rect[3]-rect[1] >= 8:
            yield rect, 1
            yield rect, -1


def wall_grip_goals(simulator, state, goal):
    """Return at most 16 real wall-contact goals, prioritizing solid movers.

    Coordinates guide search only; achieved() requires actual climb state and
    contact with the chosen face. No position, velocity or resource is assigned.
    """
    if state.dead or state.room not in simulator.map.rooms:
        return []
    room = simulator.map.rooms[state.room]
    candidates = []
    faces = []
    for identity, entity, rect in simulator._mechanism_rects(state.room, state):
        if rect[3]-rect[1] < 16:
            continue
        original = (entity.x, entity.y, entity.x+entity.width, entity.y+entity.height)
        for facing in (1, -1):
            faces.append((rect, facing, identity, entity.kind, original))
    faces.extend((rect, facing, "", "", rect) for rect, facing in _static_faces(simulator, room))
    for rect, facing, identity, kind, original in faces:
        left, top, right, bottom = rect
        x = left-4 if facing > 0 else right+4
        if not room.x+4 <= x <= room.x+room.width-4:
            continue
        if not identity and bottom-top <= 8:
            # Thin faces are climb-over milestones, not arbitrary ceiling
            # contact. Discard stair undersides with no standing headroom.
            landing_x = left+4 if facing > 0 else right-4
            if simulator.solid_at(state.room, landing_x, top, state=state):
                continue
        inset = min(12, (bottom-top)/2)
        # A one-tile lip is grabbable while the player's feet remain below
        # its bottom: the standing collider extends 11 pixels upward.
        y = (bottom+8 if not identity and bottom-top <= 8 else
             max(top+inset, min(state.y-32, bottom-inset)))
        distance = math.hypot(x-state.x, y-state.y)
        if distance > 360 or simulator.solid_at(state.room, x, y, state=state):
            continue
        identity_part = identity or f"static:{left:g}:{top:g}:{right:g}:{bottom:g}"
        target = WallGripGoal("wall_grip", state.room, x, y,
                    id=f"wall:{identity_part}:{facing}", rect=original, facing=facing,
                    identity=identity, mechanism_kind=kind, grip_offset=y-top)
        # Preserve backward wall setups: progress toward the ultimate target
        # is only a tie-breaker after reach distance and mechanism priority.
        toward_goal = math.hypot(x-goal.x, y-goal.y)
        candidates.append((0 if identity else 1, distance+toward_goal*.1, target.key, target))
    return [row[-1] for row in sorted(candidates, key=lambda row: row[:3])[:16]]


def wall_approach_macros(state, goal):
    """Relative jump/dash/grab inputs for a requested physical wall grip.

    Search still simulates and checks every frame, including intermediate goal
    hits. These longer coordinated inputs avoid losing a wall approach when
    short isolated motions are pruned before they connect with the face.
    """
    if state.dead or goal.kind != "wall_grip" or state.room != goal.room:
        return
    x, _ = goal.waypoint(state)
    approach = RIGHT if x >= state.x else LEFT
    toward_face = RIGHT if goal.facing > 0 else LEFT
    grip = GRAB | UP | toward_face
    if state.on_ground and state.vy >= 0 and state.dashes > 0:
        release = (0,) if state.prev_input & JUMP else ()
        for jump_ticks in (12, 18, 24):
            for dash_ticks in (4, 8):
                actions = (release + (JUMP | approach,)*jump_ticks
                           + (DASH | UP | approach,)*dash_ticks + (grip,)*48)
                yield actions, len(actions)
        # A vertical dash can pass a ceiling lip that rejects the diagonal
        # approach. Release horizontal movement before dashing, then steer
        # into the actual wall face; every intermediate frame is simulated.
        for jump_ticks in (12, 16, 20, 24):
            actions = (release + (JUMP | approach,)*jump_ticks + (JUMP,)*4
                       + (DASH | UP,) + (GRAB | UP,)*12 + (grip,)*48)
            yield actions, len(actions)
    elif not state.on_ground or state.vy < 0:
        for ticks in (6, 12):
            yield (grip,)*ticks, ticks


def _wall_approach_iterator(handle, simulator, origin, goal, max_frames, max_edges, stats):
    """The ordinary recipe order, with exact in-flight rolls kept across slices."""
    from .macro_continuation import MACRO_EXPANSION
    if origin.dead or goal.kind != "wall_grip" or max_frames <= 0 or max_edges <= 0:
        return []
    edges, seen = [], set()
    x, _ = goal.waypoint(origin)
    approach = RIGHT if x >= origin.x else LEFT
    toward_face = RIGHT if goal.facing > 0 else LEFT
    grip = GRAB | UP | toward_face

    def publish():
        ranked = sorted(edges,key=lambda row:(-row[1].dashes,-row[1].stamina,len(row[0]))) if grounded else edges
        handle.results = list(ranked[:max_edges])

    def roll(actions, label):
        yield MACRO_EXPANSION
        current, used, states = origin, [], [origin]
        for mask in actions[:max_frames]:
            current = simulator.step(current, mask)
            stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            used.append(mask)
            states.append(current)
            if current.dead:
                return False
            if goal.achieved(current):
                identity = replace(current, frame=0, reason="")
                if identity not in seen:
                    seen.add(identity)
                    remember_segment(origin, current, states)
                    edges.append((used, current, label))
                    publish()
                return len(edges) >= max_edges
            if current.room != origin.room:
                return False
            # Check after all per-step predicates; another coroutine may use
            # the same simulator and replace its transient event flags.
            if len(used) % 16 == 0:
                yield None
        return False

    def runup(actions):
        current = origin
        for index, mask in enumerate(actions, 1):
            current = simulator.step(current, mask)
            stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            if current.dead or current.room != origin.room:
                break
            if index % 16 == 0:
                yield None
        return current

    grounded = origin.on_ground and origin.vy >= 0
    release = (0,) if origin.prev_input & JUMP else ()
    if grounded:
        if getattr(goal, "mechanism_kind", "") == "zipMover":
            yield from roll((GRAB | approach,)*80, "wall_walk_grab")
        for ticks in (12, 18, 24, 32, 40):
            yield from roll(release+(JUMP | approach,)*ticks+(GRAB | UP | approach,)*80,
                            f"wall_jump_grab:{ticks}")
        for run_ticks in (8, 16, 24):
            if run_ticks >= max_frames:
                break
            prefix = (approach,)*run_ticks
            current = yield from runup(prefix)
            if current.dead or current.room != origin.room or not current.on_ground or current.vy < 0:
                continue
            for ticks in (12, 18, 24, 32, 40):
                yield from roll(prefix+(JUMP | approach,)*ticks+(grip,)*80,
                                f"wall_runup_jump_grab:{run_ticks}:{ticks}")
        edges.sort(key=lambda row: (-row[1].dashes, -row[1].stamina, len(row[0])))
        publish()
        if len(edges) >= max_edges:
            return edges[:max_edges]
    else:
        for ticks in (12, 32, 64, 100):
            if (yield from roll((grip,)*ticks, f"wall_climb:{ticks}")):
                return edges
        if origin.dashes > 0:
            for ticks in range(0, 41, 2):
                actions = (approach,)*ticks+(DASH | approach,)*4+(grip,)*120
                if (yield from roll(actions, f"wall_underpass:{ticks}")):
                    return edges
    if not grounded or origin.dashes <= 0:
        return edges
    for ticks in (0, 8, 16, 24, 26, 28, 32):
        prefix = (approach,)*ticks
        current = yield from runup(prefix)
        if current.dead or current.room != origin.room or not current.on_ground or current.vy < 0:
            continue
        for actions, _ in wall_approach_macros(current, goal):
            if (yield from roll(prefix+actions, f"wall_runup:{ticks}")):
                return edges
    return edges


def _macro_wall_stats(handle, stats):
    if stats is None:
        if not hasattr(handle, "_wall_stats"):
            handle._wall_stats = {}
        stats = handle._wall_stats
    stats.setdefault("expansions", 0)
    stats.setdefault("simulated_frames", 0)
    return stats


def wall_approach_edges(simulator, origin, goal, *, deadline, max_frames=180,
                        stats=None, max_edges=4, max_expansions=None, continuation=None):
    """Try a bounded set of real run-up, jump and grab arrivals.

    No-dash jump approaches precede dash approaches. Safe run-ups matter when
    the nearest wall has an overhang: jumping immediately can hit its underside
    while the same jump from the end of the launch support clears the lip.
    Every returned prefix starts at ``origin`` and preserves its complete state.
    """
    if continuation is not None:
        stats = _macro_wall_stats(continuation, stats)
        if not hasattr(continuation, "_wall_recipe_limit"):
            continuation._wall_recipe_limit = (None if max_expansions is None else
                                                max(0, max_expansions-stats["expansions"]))
        continuation.deadline = deadline
        continuation.max_expansions = max_expansions
        key = ("wall_approach", id(simulator), id(stats), origin, goal, max_frames, max_edges)
        return continuation.advance(key,
            lambda handle:_wall_approach_iterator(handle,simulator,origin,goal,max_frames,max_edges,stats),
            deadline=deadline, stats=stats, max_expansions=max_expansions,
            recipe_limit=continuation._wall_recipe_limit)
    if origin.dead or goal.kind != "wall_grip" or max_frames <= 0 or max_edges <= 0:
        return []
    stats = stats if stats is not None else {}
    edges, seen = [], set()
    x, _ = goal.waypoint(origin)
    approach = RIGHT if x >= origin.x else LEFT
    toward_face = RIGHT if goal.facing > 0 else LEFT
    grip = GRAB | UP | toward_face

    def roll(actions, label):
        if (time.monotonic() >= deadline or max_expansions is not None
                and stats.get("expansions", 0) >= max_expansions):
            return False
        stats["expansions"] = stats.get("expansions", 0)+1
        current, used, states = origin, [], [origin]
        for mask in actions[:max_frames]:
            if len(used) % 16 == 0 and time.monotonic() >= deadline:
                return False
            current = simulator.step(current, mask)
            stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            used.append(mask)
            states.append(current)
            if current.dead:
                return False
            if goal.achieved(current):
                # Different resource and momentum arrivals remain available.
                from dataclasses import replace
                key = replace(current, frame=0, reason="")
                if key not in seen:
                    seen.add(key)
                    remember_segment(origin, current, states)
                    edges.append((used, current, label))
                return len(edges) >= max_edges
            if current.room != origin.room:
                return False
        return False

    grounded = origin.on_ground and origin.vy >= 0
    release = (0,) if origin.prev_input & JUMP else ()
    if grounded:
        if getattr(goal, "mechanism_kind", "") == "zipMover":
            # A low side touch can trigger a mover without jumping. Keep it
            # alongside higher approaches: departure timing and the opening
            # below a rising platform can matter more than initial height.
            roll((GRAB | approach,)*80, "wall_walk_grab")
        # A real run-up can catch the wall near the jump apex instead of
        # spending most stamina climbing from its bottom. Gather this small
        # no-dash family before selecting arrivals; the first slow contact
        # must not hide a later arrival carrying much more stamina.
        for ticks in (12, 18, 24, 32, 40):
            actions = release+(JUMP | approach,)*ticks+(GRAB | UP | approach,)*80
            roll(actions, f"wall_jump_grab:{ticks}")
        for run_ticks in (8, 16, 24):
            if (run_ticks >= max_frames or time.monotonic() >= deadline or max_expansions is not None
                    and stats.get("expansions", 0) >= max_expansions):
                break
            prefix = (approach,)*run_ticks
            current = origin
            for mask in prefix:
                if time.monotonic() >= deadline:
                    break
                current = simulator.step(current, mask)
                stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
                if current.dead or current.room != origin.room:
                    break
            if (current.dead or current.room != origin.room or not current.on_ground
                    or current.vy < 0 or time.monotonic() >= deadline):
                continue
            for ticks in (12, 18, 24, 32, 40):
                roll(prefix+(JUMP | approach,)*ticks+(grip,)*80,
                     f"wall_runup_jump_grab:{run_ticks}:{ticks}")
        edges.sort(key=lambda row: (-row[1].dashes, -row[1].stamina, len(row[0])))
        if len(edges) >= max_edges:
            return edges[:max_edges]
    else:
        for ticks in (12, 32, 64, 100):
            if roll((grip,)*ticks, f"wall_climb:{ticks}"):
                return edges
        # Releasing an outside wall before a horizontal dash can pass below
        # a column's bottom and reach the opposite side of a narrow shaft.
        # The delay is a search branch, never a geometry teleport or drop.
        if origin.dashes > 0:
            for ticks in range(0, 41, 2):
                actions = (approach,)*ticks+(DASH | approach,)*4+(grip,)*120
                if roll(actions, f"wall_underpass:{ticks}"):
                    return edges
    if not grounded or origin.dashes <= 0:
        return edges
    # These are actual walking prefixes, including ledge departure. A prefix
    # which leaves the support simply cannot start a grounded jump macro.
    for ticks in (0, 8, 16, 24, 26, 28, 32):
        if (time.monotonic() >= deadline or max_expansions is not None
                and stats.get("expansions", 0) >= max_expansions):
            break
        prefix = (approach,)*ticks
        current = origin
        for mask in prefix:
            current = simulator.step(current, mask)
            stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            if current.dead or current.room != origin.room:
                break
        if current.dead or current.room != origin.room or not current.on_ground or current.vy < 0:
            continue
        for actions, _ in wall_approach_macros(current, goal):
            if roll(prefix+actions, f"wall_runup:{ticks}"):
                return edges
            if time.monotonic() >= deadline:
                break
    return edges


def _wall_relay_iterator(handle, simulator, origin, goal, max_frames, max_edges, stats):
    """Keep setup, child macro, child search and final replay as separate stages."""
    from .macro_continuation import MacroContinuation, MACRO_PAUSE
    from .planner import SearchContinuation, search_leg
    if (origin.dead or goal.kind != "wall_grip" or getattr(goal, "identity", "")
            or goal.room != origin.room or goal.y >= origin.y-12
            or max_frames <= 0 or max_edges <= 0):
        return []

    class NoDash:
        def __init__(self):
            self.last_step_started_dash = False
            self._planning_identity = object()
        def __getattr__(self, name):
            return getattr(simulator, name)
        def step(self, before, mask):
            self.last_step_started_dash = False
            if before.frame-origin.frame >= max_frames:
                return replace(before, dead=True, reason="wall_relay_budget")
            after = simulator.step(before, mask)
            self.last_step_started_dash = bool(getattr(simulator,"last_step_started_dash",False))
            if mask & DASH or self.last_step_started_dash:
                return replace(after, dead=True, reason="wall_relay_dash")
            return after

    wrapped = NoDash()
    setups = [setup for setup in wall_grip_goals(wrapped, origin, goal)
              if (not setup.identity and setup.facing != goal.facing and setup.rect != goal.rect
                  and setup.y >= goal.y+12 and setup.y < origin.y-4
                  and abs(setup.x-goal.x) <= 64
                  and math.hypot(setup.x-origin.x,setup.y-origin.y) <= 160)]
    setups.sort(key=lambda setup: (math.hypot(setup.x-origin.x,setup.y-origin.y)
                                  + abs(setup.x-goal.x)*.5, setup.key))
    edges, seen, consumed = [], set(), 0

    def available():
        ceiling = handle.max_expansions
        if ceiling is None:
            ceiling = float("inf")
        return max(0, min(320-consumed, ceiling-stats["expansions"],
                          stats.get("max_expansions",float("inf"))-stats["expansions"]))

    for setup in setups[:2]:
        if consumed >= 320:
            break
        child = MacroContinuation(handle.node_budget)
        handle.children.append(child)
        while not child.done:
            if available() <= 0:
                if consumed >= 320:
                    return edges
                yield MACRO_PAUSE
                continue
            before = stats["expansions"]
            wall_approach_edges(wrapped, origin, setup, deadline=handle.deadline,
                max_frames=max_frames, stats=stats, max_edges=2, continuation=child,
                max_expansions=stats["expansions"]+int(min(32-child.expansions,available())))
            consumed += stats["expansions"]-before
            if not child.done:
                yield MACRO_PAUSE
        approaches = list(child.results)
        child.release()
        for prefix, arrival, label in approaches:
            remaining = max_frames-len(prefix)
            if remaining <= 0 or consumed >= 320:
                break
            search = SearchContinuation(handle.node_budget)
            handle.children.append(search)
            used_expansions = 0
            result = None
            while used_expansions < 128 and consumed < 320:
                if available() <= 0:
                    yield MACRO_PAUSE
                    continue
                result = search_leg(wrapped, arrival, goal, deadline=handle.deadline,
                    max_expansions=int(min(64,128-used_expansions,available())),
                    max_frames=remaining, variant=4, allow_unsupported=False,
                    continuation=search)
                stats["expansions"] += result.expansions
                stats["simulated_frames"] += result.simulated_frames
                consumed += result.expansions
                used_expansions += result.expansions
                if result.reached or result.reason not in ("budget","time_budget","expansion_budget"):
                    break
                yield MACRO_PAUSE
            search.release()
            if result is None or not result.reached:
                continue
            actions = list(prefix)+list(result.actions)
            current, states, valid = origin, [origin], True
            for index, mask in enumerate(actions, 1):
                current = wrapped.step(current, mask)
                stats["simulated_frames"] += 1
                states.append(current)
                if current.dead or current.room != origin.room:
                    valid = False
                    break
                # A bounded, already computed suffix must finish its proof
                # even on the last search expansion. This adds no new branch.
            if not valid or not goal.achieved(current) or current != result.state:
                continue
            identity = replace(current, frame=0, reason="")
            if identity in seen:
                continue
            seen.add(identity)
            remember_segment(origin, current, states)
            edges.append((actions,current,f"wall_relay:{setup.key}:{label}"))
            handle.results = list(edges)
            if len(edges) >= max_edges:
                return edges
    return edges


def wall_relay_approach_edges(simulator, origin, goal, *, deadline, max_frames=220,
                              stats=None, max_expansions=None, max_edges=2, continuation=None):
    """Reach a higher static wall through a lower real grip without dashing.

    Nearby opposite faces can share a narrow gap. A jump from the lower face
    reaches an upper face that is hard to catch directly from the floor. Both
    legs keep the caller's simulator/constraint wrapper and every input; only
    a physically reached final grip is returned. No recursive relay search.
    """
    if continuation is not None:
        stats = _macro_wall_stats(continuation, stats)
        continuation.deadline = deadline
        continuation.max_expansions = max_expansions
        key = ("wall_relay",id(simulator),id(stats),origin,goal,max_frames,max_edges)
        return continuation.advance(key,
            lambda handle:_wall_relay_iterator(handle,simulator,origin,goal,max_frames,max_edges,stats),
            deadline=deadline, stats=stats, max_expansions=max_expansions, recipe_limit=320)
    if (origin.dead or goal.kind != "wall_grip" or getattr(goal, "identity", "")
            or goal.room != origin.room or goal.y >= origin.y-12
            or max_frames <= 0 or max_edges <= 0 or time.monotonic() >= deadline):
        return []
    stats = stats if stats is not None else {}
    stats.setdefault("expansions", 0)
    stats.setdefault("simulated_frames", 0)
    limit = min(max_expansions if max_expansions is not None else float("inf"),
                stats.get("max_expansions", float("inf")), stats["expansions"]+320)
    if stats["expansions"] >= limit:
        return []

    class NoDash:
        def __init__(self):
            self.last_step_started_dash = False
        def __getattr__(self, name):
            return getattr(simulator, name)
        def step(self, before, mask):
            self.last_step_started_dash = False
            if time.monotonic() >= deadline or before.frame-origin.frame >= max_frames:
                return replace(before, dead=True, reason="wall_relay_budget")
            after = simulator.step(before, mask)
            self.last_step_started_dash = bool(getattr(simulator,"last_step_started_dash",False))
            if mask & DASH or self.last_step_started_dash:
                return replace(after, dead=True, reason="wall_relay_dash")
            return after

    wrapped = NoDash()
    setups = [setup for setup in wall_grip_goals(wrapped, origin, goal)
              if (not setup.identity and setup.facing != goal.facing and setup.rect != goal.rect
                  and setup.y >= goal.y+12 and setup.y < origin.y-4
                  and abs(setup.x-goal.x) <= 64
                  and math.hypot(setup.x-origin.x,setup.y-origin.y) <= 160)]
    setups.sort(key=lambda setup: (math.hypot(setup.x-origin.x,setup.y-origin.y)
                                  + abs(setup.x-goal.x)*.5, setup.key))
    edges, seen = [], set()
    from .planner import search_leg
    for setup in setups[:2]:
        if time.monotonic() >= deadline or stats["expansions"] >= limit:
            break
        approaches = wall_approach_edges(wrapped, origin, setup, deadline=deadline,
            max_frames=max_frames, stats=stats, max_edges=2,
            max_expansions=min(limit, stats["expansions"]+32))
        for prefix, arrival, label in approaches:
            remaining = max_frames-len(prefix)
            if remaining <= 0 or stats["expansions"] >= limit or time.monotonic() >= deadline:
                break
            result = search_leg(wrapped, arrival, goal, deadline=deadline,
                max_expansions=int(min(128,limit-stats["expansions"])),
                max_frames=remaining, variant=4, allow_unsupported=False)
            stats["expansions"] += result.expansions
            stats["simulated_frames"] += result.simulated_frames
            if not result.reached:
                continue
            actions = list(prefix)+list(result.actions)
            current, states, valid = origin, [origin], True
            for mask in actions:
                current = wrapped.step(current, mask)
                stats["simulated_frames"] += 1
                states.append(current)
                if current.dead or current.room != origin.room:
                    valid = False
                    break
            if not valid or not goal.achieved(current) or current != result.state:
                continue
            key = replace(current, frame=0, reason="")
            if key in seen:
                continue
            seen.add(key)
            remember_segment(origin, current, states)
            edges.append((actions,current,f"wall_relay:{setup.key}:{label}"))
            if len(edges) >= max_edges:
                return edges
    return edges


def wall_launch_edges(simulator, grip_state, max_frames=140):
    """Bounded climb/reverse-jump rollouts ending on an alive real landing.

    Every returned edge contains every input, including the neutral update that
    allows native grounded dash restoration. Room changes and death reject the
    edge; transient mid-air progress is never reported as a launch support.
    """
    if (grip_state.dead or grip_state.state != 1 or grip_state.stamina <= 0
            or max_frames < 2):
        return []
    room = grip_state.room
    away = LEFT if grip_state.facing > 0 else RIGHT
    edges = []
    for hold in (0, 8, 16, 24, 32, 48):
        if hold+2 > max_frames:
            continue
        state = grip_state
        actions, states = [], [state]

        def step(mask):
            nonlocal state
            if len(actions) >= max_frames:
                return False
            state = simulator.step(state, mask)
            actions.append(mask)
            states.append(state)
            return not state.dead and state.room == room

        valid = True
        # Even the zero-climb branch must release an earlier jump button.
        ticks = max(hold, int(bool(grip_state.prev_input & JUMP)))
        for _ in range(ticks):
            if not step(GRAB | UP):
                valid = False
                break
        if not valid:
            continue
        landed = state.on_ground and state.vy >= 0
        if not landed:
            if state.state != 1:
                continue
            for _ in range(min(70, max_frames-len(actions)-1)):
                if not step(JUMP | away):
                    valid = False
                    break
                if state.on_ground and state.vy >= 0:
                    landed = True
                    break
        if not valid or not landed or not step(0):
            continue
        if not state.on_ground or state.vy < 0:
            continue
        label = f"wall_reverse:{grip_state.facing}:climb{hold}"
        remember_segment(grip_state, state, states)
        edges.append((actions.copy(), state, label))
    return edges
