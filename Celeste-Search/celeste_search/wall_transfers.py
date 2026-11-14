"""Bounded jumps from actual static-wall contacts into a requested goal.

Climbing and launch timing are ordinary simulated inputs. No wall waypoint
changes the player's position, stamina, dash count, or public route progress.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import chain, islice
import time

from .physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment
from .wall_moves import _static_faces


def _static_contact(simulator, origin):
    if (origin.dead or origin.state != 1 or origin.stamina <= 0
            or origin.facing not in (-1, 1) or not hasattr(simulator, "_tiles")):
        return None
    data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
    room = data.rooms.get(origin.room)
    if room is None or simulator.solid_at(origin.room, origin.x, origin.y,
                                         ducking=origin.ducking, state=origin):
        return None
    height = 6 if origin.ducking else 11
    for rect, facing in _static_faces(simulator, room):
        left, top, right, bottom = rect
        side_x = left-4 if facing > 0 else right+4
        if (facing == origin.facing and abs(origin.x-side_x) < .1
                and origin.y > top and origin.y-height < bottom
                and simulator._wall(origin.room, origin.x, origin.y, facing, 1)):
            return rect
    return None


def _recipes(origin, target, allow_dash):
    waypoint = getattr(target, "waypoint", None)
    x, _ = waypoint(origin) if waypoint is not None else (target.x, target.y)
    toward = RIGHT if x >= origin.x else LEFT
    away = LEFT if origin.facing > 0 else RIGHT
    directions = tuple(dict.fromkeys((toward, 0, away)))
    face = (RIGHT if target.facing > 0 else LEFT) if target.kind == "wall_grip" else toward
    catch = GRAB | UP | face
    # A held Jump must be released through a real climbing update before the
    # next jump can fire. The climb prefix itself provides that release.
    holds = (0, 8, 12, 16, 24)
    for delay in (12, 28):
        for hold in holds:
            prefix = (GRAB | UP,)*max(hold, int(bool(origin.prev_input & JUMP)))
            for direction in directions:
                yield (prefix+(JUMP | direction,)*delay+(catch,)*140,
                       f"climb_jump:{hold}:{direction}:{delay}")
    if not allow_dash or origin.dashes <= 0:
        return
    # Interleave climb heights before trying every aim: a bounded caller can
    # reach the useful late-dash family without exhausting all zero-hold cases.
    for delay in (28, 36, 20, 12):
        for direction in directions:
            for hold in holds:
                prefix = (GRAB | UP,)*max(hold, int(bool(origin.prev_input & JUMP)))
                for aim in (UP | toward, toward, UP):
                    yield (prefix+(JUMP | direction,)*delay+(DASH | aim,)*6+(catch,)*140,
                           f"climb_jump_dash:{hold}:{direction}:{delay}:{aim}")


@dataclass(frozen=True)
class _RegripRecipe:
    facing: int
    climb_ticks: int
    jump_hold: int | None = None
    finish_height: float | None = None

    def inputs(self, simulator, current):
        """Repeat only after a real catch; a failed catch ends this branch.

        Releasing Grab before a directionless jump invokes the ordinary wall
        jump rather than the stamina-consuming climb jump. Its horizontal
        movement remains entirely simulated, including collisions at steps.
        """
        toward = RIGHT if self.facing > 0 else LEFT
        while True:
            for _ in range(self.climb_ticks):
                yield GRAB | UP | toward
            # Preserve efficient full jumps below the final approach. A short
            # jump near the guide can catch the wall before rising across an
            # exit at the wrong horizontal position. This only selects inputs;
            # the caller's strict goal and constraints still decide success.
            short = (self.jump_hold is not None and
                     (self.finish_height is None or current().y <= self.finish_height+48))
            yield 0
            yield JUMP
            for tick in range(80):
                # jump_hold includes the directionless first JUMP above.
                jump = JUMP if not short or tick+1 < self.jump_hold else 0
                yield jump | GRAB | UP | toward
                state = current()
                if (state.state == 1 and state.facing == self.facing
                        and _static_contact(simulator, state) is not None):
                    if short:
                        # This catch aligns x with the wall. Climb physically
                        # toward the target instead of immediately jumping off
                        # again; stamina, death and frame caps remain active.
                        while True:
                            yield GRAB | UP | toward
                    break
            else:
                return


def static_wall_transfer_edges(simulator, origin, target, *, deadline,
                               max_frames=220, stats=None, max_expansions=None,
                               max_edges=4, continuation=None, remember_results=True):
    """Return continuous ``(actions, state, label)`` static-wall transfers.

    Supports, real wall grips and public-hit wrappers use their own strict
    achieved predicate. Every step goes through the caller's simulator, so
    route restrictions and resource/phase state cannot be bypassed.
    """
    if continuation is not None:
        return _continued_transfers(simulator,origin,target,deadline=deadline,
            max_frames=max_frames,stats=stats,max_expansions=max_expansions,
            max_edges=max_edges,continuation=continuation,remember_results=remember_results)
    if max_frames <= 0 or max_edges <= 0 or time.monotonic() >= deadline:
        return []
    stats = stats if stats is not None else {}
    limit = min(max_expansions if max_expansions is not None else float("inf"),
                stats.get("max_expansions", float("inf")),
                stats.get("expansions", 0)+210)
    if stats.get("expansions", 0) >= limit or _static_contact(simulator, origin) is None:
        return []
    if target.achieved(origin):
        return [([], origin, "static_wall_transfer:already_reached:dash_starts=0")]
    allow_dash = not getattr(simulator, "no_dash", False)
    seen, recipes_seen = {}, set()
    face = target.facing if target.kind == "wall_grip" else origin.facing
    waypoint = getattr(target, "waypoint", None)
    finish_height = waypoint(origin)[1] if waypoint is not None else target.y
    regrips = chain(
        ((_RegripRecipe(face, climb), f"release_regrip:{climb}")
         for climb in (0, 4, 8)),
        ((_RegripRecipe(face, climb, held, finish_height), f"release_regrip:{climb}:held{held}")
         for held in (8, 4) for climb in (0, 4, 8)))
    for recipe, label in chain(regrips, _recipes(origin, target, allow_dash)):
        if time.monotonic() >= deadline or stats.get("expansions", 0) >= limit:
            break
        if not isinstance(recipe, _RegripRecipe):
            recipe = recipe[:max_frames]
        if recipe in recipes_seen:
            continue
        recipes_seen.add(recipe)
        stats["expansions"] = stats.get("expansions", 0)+1
        state, actions, states, dash_starts = origin, [], [origin], []
        inputs = (recipe.inputs(simulator, lambda: state)
                  if isinstance(recipe, _RegripRecipe) else recipe)
        for mask in islice(inputs, max_frames):
            if time.monotonic() >= deadline:
                break
            state = simulator.step(state, mask)
            started_dash = bool(getattr(simulator, "last_step_started_dash", False))
            if started_dash:
                dash_starts.append(len(actions)+1)
            stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
            actions.append(mask)
            states.append(state)
            if state.dead or not allow_dash and (mask & DASH or started_dash):
                break
            if target.achieved(state):
                key = replace(state, frame=0, reason="")
                rank = (len(dash_starts), -state.dashes, -state.stamina,
                        len(actions), tuple(-tick for tick in dash_starts))
                if key not in seen or rank < seen[key][0]:
                    seen[key] = (rank, actions, state, label, states)
                break
            if state.room != origin.room:
                break
    chosen, deferred, bands = [], [], set()
    for row in sorted(seen.values(), key=lambda row:row[0]):
        rank, _, state, _, _ = row
        band = (rank[0], state.dashes, state.state, round(state.y/8),
                round(state.vx/45), round(state.vy/45), int(state.stamina/20))
        if band in bands:
            deferred.append(row)
        else:
            chosen.append(row)
            bands.add(band)
        if len(chosen) >= max_edges:
            break
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    result = []
    for rank, actions, state, label, states in sorted(chosen,key=lambda row:row[0]):
        if remember_results:
            remember_segment(origin,state,states)
        result.append((actions,state,f"static_wall_transfer:{label}:dash_starts={rank[0]}"))
    return result


def _transfer_results(rows, max_edges):
    chosen, deferred, bands = [], [], set()
    for row in sorted(rows,key=lambda row:row[0]):
        rank,_,state,_,_ = row
        band=(rank[0],state.dashes,state.state,round(state.y/8),
              round(state.vx/45),round(state.vy/45),int(state.stamina/20))
        if band in bands:
            deferred.append(row)
        else:
            chosen.append(row)
            bands.add(band)
        if len(chosen)>=max_edges:
            break
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    return [(actions,state,f"static_wall_transfer:{label}:dash_starts={rank[0]}")
            for rank,actions,state,label,_ in sorted(chosen,key=lambda row:row[0])]


def _continued_transfers(simulator,origin,target,*,deadline,max_frames,stats,
                         max_expansions,max_edges,continuation,remember_results):
    from .macro_continuation import MACRO_EXPANSION
    if stats is None:
        stats=continuation.__dict__.setdefault("stats",{})
    key=(id(simulator),id(stats),origin,target,max_frames,max_edges)
    def recipes(work):
        if (max_frames<=0 or max_edges<=0 or _static_contact(simulator,origin) is None):
            return []
        if target.achieved(origin):
            return [([],origin,"static_wall_transfer:already_reached:dash_starts=0")]
        allow_dash=not getattr(simulator,"no_dash",False)
        seen,recipes_seen={},set()
        face=target.facing if target.kind=="wall_grip" else origin.facing
        waypoint=getattr(target,"waypoint",None)
        finish_height=waypoint(origin)[1] if waypoint is not None else target.y
        regrips=chain(
            ((_RegripRecipe(face,climb),f"release_regrip:{climb}") for climb in (0,4,8)),
            ((_RegripRecipe(face,climb,held,finish_height),f"release_regrip:{climb}:held{held}")
             for held in (8,4) for climb in (0,4,8)))
        for recipe,label in chain(regrips,_recipes(origin,target,allow_dash)):
            if not isinstance(recipe,_RegripRecipe):
                recipe=recipe[:max_frames]
            if recipe in recipes_seen:
                continue
            recipes_seen.add(recipe)
            yield MACRO_EXPANSION
            state,actions,states,dash_starts=origin,[],[origin],[]
            inputs=recipe.inputs(simulator,lambda:state) if isinstance(recipe,_RegripRecipe) else recipe
            for mask in islice(inputs,max_frames):
                if len(actions)%16==0:
                    yield None
                state=simulator.step(state,mask)
                started_dash=bool(getattr(simulator,"last_step_started_dash",False))
                if started_dash:
                    dash_starts.append(len(actions)+1)
                stats["simulated_frames"]=stats.get("simulated_frames",0)+1
                actions.append(mask)
                states.append(state)
                if state.dead or not allow_dash and (mask&DASH or started_dash):
                    break
                if target.achieved(state):
                    identity=replace(state,frame=0,reason="")
                    rank=(len(dash_starts),-state.dashes,-state.stamina,
                          len(actions),tuple(-tick for tick in dash_starts))
                    if identity not in seen or rank<seen[identity][0]:
                        seen[identity]=(rank,actions,state,label,())
                        # Suspended histories are separate from graph nodes;
                        # bound their count while preserving ranked diversity.
                        if len(seen)>64:
                            chosen_ids={replace(row[1],frame=0,reason="")
                                        for row in _transfer_results(seen.values(),max_edges*4)}
                            ranked=sorted(seen,key=lambda item:(item not in chosen_ids,seen[item][0]))
                            seen={item:seen[item] for item in ranked[:64]}
                        # A caller that certifies/adopts these arrivals can
                        # publish their exact prefix once, after selection.
                        if remember_results:
                            remember_segment(origin,state,states)
                        work.results=_transfer_results(seen.values(),max_edges)
                    break
                if state.room!=origin.room:
                    break
        return work.results
    limit=min(value for value in (max_expansions,stats.get("max_expansions")) if value is not None) \
        if max_expansions is not None or stats.get("max_expansions") is not None else None
    recipe_limit=continuation.__dict__.setdefault("recipe_limit",
        min(210,max(0,limit-stats.get("expansions",0))) if limit is not None else 210)
    return continuation.advance(key,recipes,deadline=deadline,stats=stats,
        max_expansions=limit,recipe_limit=recipe_limit)
